import torch
from torch import nn
import torch.nn.functional as F


class nnUNet(nn.Module):
    @staticmethod
    def physics_block(in_channels, out_channels):
        fcl = torch.nn.Sequential(torch.nn.Linear(in_channels, out_channels),
                                  torch.nn.SELU(),
                                  torch.nn.Linear(out_channels, out_channels),
                                  torch.nn.SELU(),
                                  )
        return fcl

    @staticmethod
    def tiling_and_concat(fcl_input, conv_input, concat_axis=1):
        # Expand dimensions N times until it matches that of the conv it will be concatenated to
        expanded_input = fcl_input[..., None, None, None]
        # Tile across all dimensions EXCEPT dimension being concatenated to AND batch dimension: First + Second one
        tiled_fcl = expanded_input.repeat((1, 1,) + conv_input.shape[concat_axis + 1:])
        physics_concat = torch.cat([conv_input, tiled_fcl], dim=concat_axis)
        return physics_concat

    @staticmethod
    def contracting_block(in_channels, out_channels, kernel_size=3):
        """
        This function creates one contracting block
        """
        block = torch.nn.Sequential(
            torch.nn.Conv3d(kernel_size=kernel_size, in_channels=in_channels, out_channels=out_channels, padding=1),
            torch.nn.LeakyReLU(),
            torch.nn.Conv3d(kernel_size=kernel_size, in_channels=out_channels, out_channels=out_channels, padding=1),
            torch.nn.LeakyReLU(),
        )
        return block

    @staticmethod
    def expansive_block(in_channels, mid_channel, final_channel, kernel_size=3):
        """
        This function creates one expansive block
        """
        block = torch.nn.Sequential(
            torch.nn.Conv3d(kernel_size=kernel_size, in_channels=in_channels, out_channels=mid_channel, padding=1),
            torch.nn.LeakyReLU(),
            torch.nn.Conv3d(kernel_size=kernel_size, in_channels=mid_channel, out_channels=final_channel, padding=1),
            torch.nn.LeakyReLU(),
            torch.nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True))
        # )
        return block

    @staticmethod
    def penultimate_block(in_channels, mid_channel, kernel_size=3):
        """
        This returns final block
        """
        block = torch.nn.Sequential(
            torch.nn.Conv3d(kernel_size=kernel_size, in_channels=in_channels, out_channels=mid_channel, padding=1),
            torch.nn.LeakyReLU(),
            torch.nn.Conv3d(kernel_size=kernel_size, in_channels=mid_channel, out_channels=mid_channel, padding=1),
            torch.nn.LeakyReLU(),
        )
        return block

    @staticmethod
    def final_block(mid_channel, out_channels, kernel_size=3):
        """
        This returns final block
        """
        block = torch.nn.Sequential(
            torch.nn.Conv3d(kernel_size=kernel_size, in_channels=mid_channel, out_channels=out_channels, padding=1),
            torch.nn.LeakyReLU(),
        )
        return block

    def __init__(self, in_channel, out_channel, physics_flag=False, physics_input=None, physics_output=0):
        # Encode
        super(nnUNet, self).__init__()

        # Physics
        self.physics_flag = physics_flag

        if not physics_flag:
            physics_output = 0
        self.conv_encode1 = self.contracting_block(in_channels=in_channel, out_channels=30)
        self.conv_maxpool1 = torch.nn.MaxPool3d(kernel_size=2)
        self.conv_encode2 = self.contracting_block(30 + physics_output, 60)
        self.conv_maxpool2 = torch.nn.MaxPool3d(kernel_size=2)
        self.conv_encode3 = self.contracting_block(60, 120)
        self.conv_maxpool3 = torch.nn.MaxPool3d(kernel_size=2)
        self.conv_encode4 = self.contracting_block(120, 240)
        self.conv_maxpool4 = torch.nn.MaxPool3d(kernel_size=2)

        if physics_flag:
            self.phys = self.physics_block(physics_input, physics_output)

        # Bottleneck
        self.bottleneck = torch.nn.Sequential(
            torch.nn.Conv3d(kernel_size=3, in_channels=240, out_channels=480, padding=1),
            torch.nn.LeakyReLU(),
            torch.nn.Conv3d(kernel_size=3, in_channels=480, out_channels=240, padding=1),
            torch.nn.LeakyReLU(),
            torch.nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        )
        # Decode
        self.conv_decode3 = self.expansive_block(480, 240, 120)
        self.conv_decode2 = self.expansive_block(240, 120, 60)
        self.conv_decode1 = self.expansive_block(120, 60, 30)
        self.penultimate_layer = self.penultimate_block(60 + physics_output, 30)
        self.final_layer = self.final_block(30, out_channel)

    @staticmethod
    def crop_and_concat(upsampled, bypass, crop=False):
        """
        This layer crop the layer from contraction block and concat it with expansive block vector
        """
        if crop:
            c = (bypass.size()[2] - upsampled.size()[2]) // 2
            bypass = F.pad(bypass, (-c, -c, -c, -c))
        return torch.cat((upsampled, bypass), 1)

    def forward(self, x, physics=None):
        # Physics
        if self.physics_flag:
            physics_block = self.phys(physics)
        # Encode
        encode_block1 = self.conv_encode1(x)
        encode_pool1 = self.conv_maxpool1(encode_block1)

        if self.physics_flag:
            encode_pool1 = self.tiling_and_concat(physics_block, encode_pool1)

        encode_block2 = self.conv_encode2(encode_pool1)
        encode_pool2 = self.conv_maxpool2(encode_block2)
        encode_block3 = self.conv_encode3(encode_pool2)
        encode_pool3 = self.conv_maxpool3(encode_block3)
        encode_block4 = self.conv_encode4(encode_pool3)
        encode_pool4 = self.conv_maxpool4(encode_block4)
        # Bottleneck
        bottleneck1 = self.bottleneck(encode_pool4)
        # Decode: Start with concat
        decode_block4 = self.crop_and_concat(bottleneck1, encode_block4, crop=False)
        cat_layer3 = self.conv_decode3(decode_block4)
        decode_block3 = self.crop_and_concat(cat_layer3, encode_block3, crop=False)
        cat_layer2 = self.conv_decode2(decode_block3)
        decode_block2 = self.crop_and_concat(cat_layer2, encode_block2, crop=False)
        cat_layer1 = self.conv_decode1(decode_block2)
        decode_block1 = self.crop_and_concat(cat_layer1, encode_block1, crop=False)

        if self.physics_flag:
            decode_block1 = self.tiling_and_concat(physics_block, decode_block1)

        features = self.penultimate_layer(decode_block1)
        final_layer = self.final_layer(features)

        return final_layer, features
