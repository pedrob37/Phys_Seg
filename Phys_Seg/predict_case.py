import torch
import numpy as np
from typing import Callable, Union
import torch.nn.functional as F
from monai.data.utils import compute_importance_map, dense_patch_slices, get_valid_patch_size
from monai.utils import BlendMode, PytorchPadMode, fall_back_tuple


# def pad_patient_3D(patient, shape_must_be_divisible_by=16, min_size=None):
#     if not (isinstance(shape_must_be_divisible_by, list) or isinstance(shape_must_be_divisible_by, tuple)):
#         shape_must_be_divisible_by = [shape_must_be_divisible_by] * 3
#     shp = patient.shape
#     new_shp = [shp[0] + shape_must_be_divisible_by[0] - shp[0] % shape_must_be_divisible_by[0],
#                shp[1] + shape_must_be_divisible_by[1] - shp[1] % shape_must_be_divisible_by[1],
#                shp[2] + shape_must_be_divisible_by[2] - shp[2] % shape_must_be_divisible_by[2]]
#     for i in range(len(shp)):
#         if shp[i] % shape_must_be_divisible_by[i] == 0:
#             new_shp[i] -= shape_must_be_divisible_by[i]
#     if min_size is not None:
#         new_shp = np.max(np.vstack((np.array(new_shp), np.array(min_size))), 0)
#     return reshape_by_padding_upper_coords(patient, new_shp, 0), shp


# def reshape_by_padding_upper_coords(image, new_shape, pad_value=None):
#     shape = tuple(list(image.shape))
#     new_shape = tuple(np.max(np.concatenate((shape, new_shape)).reshape((2,len(shape))), axis=0))
#     if pad_value is None:
#         if len(shape) == 2:
#             pad_value = image[0,0]
#         elif len(shape) == 3:
#             pad_value = image[0, 0, 0]
#         else:
#             raise ValueError("Image must be either 2 or 3 dimensional")
#     res = np.ones(list(new_shape), dtype=image.dtype) * pad_value
#     if len(shape) == 2:
#         res[0:0+int(shape[0]), 0:0+int(shape[1])] = image
#     elif len(shape) == 3:
#         res[0:0+int(shape[0]), 0:0+int(shape[1]), 0:0+int(shape[2])] = image
#     return res
def image_preprocessing(patient_data):
    if len(patient_data.shape) < 5:
        shape_mismatch = 5 - len(patient_data.shape)
        patient_data = patient_data[(*([None] * shape_mismatch), ...)]
    return (patient_data - np.mean(patient_data)) / np.std(patient_data)


def physics_preprocessing(physics_input, experiment_type):
    physics_input = torch.from_numpy(physics_input[None, :])
    if experiment_type == 'MPRAGE':
        TI_physics = physics_input[:, 0]
        # print(physics_input.shape)
        TR_physics = physics_input[:, 0] + physics_input[:, 1]
        TI_expo_physics = torch.exp(-physics_input[:, 0])
        TR_expo_physics = torch.exp(-physics_input[:, 0] - physics_input[:, 1])
        overall_physics = torch.cat((torch.stack((TI_physics,
                                                  TR_physics), dim=1),
                                     torch.stack((TI_expo_physics,
                                                  TR_expo_physics), dim=1)), dim=1)
    elif experiment_type == 'SPGR':
        TR_expo_params = torch.unsqueeze(torch.exp(-physics_input[:, 0]), dim=1)
        TE_expo_params = torch.unsqueeze(torch.exp(-physics_input[:, 1]), dim=1)
        FA_sin_params = torch.unsqueeze(torch.sin(physics_input[:, 2] * 3.14159265 / 180), dim=1)
        overall_physics = torch.cat((physics_input, torch.stack((TR_expo_params, TE_expo_params, FA_sin_params), dim=1).squeeze()), dim=1)
    return overall_physics.float()


def custom_sliding_window_inference(
    inputs: Union[torch.Tensor, tuple],
    roi_size,
    sw_batch_size: int,
    predictor: Callable,
    overlap: float = 0.25,
    mode: Union[BlendMode, str] = BlendMode.CONSTANT,
    padding_mode: Union[PytorchPadMode, str] = PytorchPadMode.CONSTANT,
    cval=0,
    uncertainty_flag=False,
    num_loss_passes=20
):
    """
    Sliding window inference on `inputs` with `predictor`.

    When roi_size is larger than the inputs' spatial size, the input image are padded during inference.
    To maintain the same spatial sizes, the output image will be cropped to the original input size.

    Args:
        inputs: input image to be processed (assuming NCHW[D])
        roi_size (list, tuple): the spatial window size for inferences.
            When its components have None or non-positives, the corresponding inputs dimension will be used.
            if the components of the `roi_size` are non-positive values, the transform will use the
            corresponding components of img size. For example, `roi_size=(32, -1)` will be adapted
            to `(32, 64)` if the second spatial dimension size of img is `64`.
        sw_batch_size: the batch size to run window slices.
        predictor: given input tensor `patch_data` in shape NCHW[D], `predictor(patch_data)`
            should return a prediction with the same spatial shape and batch_size, i.e. NMHW[D];
            where HW[D] represents the patch spatial size, M is the number of output channels, N is `sw_batch_size`.
        overlap: Amount of overlap between scans.
        mode: {``"constant"``, ``"gaussian"``}
            How to blend output of overlapping windows. Defaults to ``"constant"``.

            - ``"constant``": gives equal weight to all predictions.
            - ``"gaussian``": gives less weight to predictions on edges of windows.

        padding_mode: {``"constant"``, ``"reflect"``, ``"replicate"``, ``"circular"``}
            Padding mode when ``roi_size`` is larger than inputs. Defaults to ``"constant"``
            See also: https://pytorch.org/docs/stable/nn.functional.html#pad
        cval: fill value for 'constant' padding mode. Default: 0

    Raises:
        NotImplementedError: inputs must have batch_size=1.

    Note:
        - input must be channel-first and have a batch dim, support both spatial 2D and 3D.
        - currently only supports `inputs` with batch_size=1.
    """
    assert 0 <= overlap < 1, "overlap must be >= 0 and < 1."

    # determine image spatial size and batch size
    # Note: all input images must have the same image size and batch size
    inputs_type = type(inputs)
    if inputs_type == tuple:
        phys_inputs = inputs[1]
        inputs = inputs[0]
    num_spatial_dims = len(inputs.shape) - 2
    image_size_ = list(inputs.shape[2:])
    batch_size = inputs.shape[0]

    # TODO: Enable batch sizes > 1 in future
    if batch_size > 1:
        raise NotImplementedError("inputs must have batch_size=1.")

    roi_size = fall_back_tuple(roi_size, image_size_)
    # in case that image size is smaller than roi size
    image_size = tuple(max(image_size_[i], roi_size[i]) for i in range(num_spatial_dims))
    pad_size = []
    for k in range(len(inputs.shape) - 1, 1, -1):
        diff = max(roi_size[k - 2] - inputs.shape[k], 0)
        half = diff // 2
        pad_size.extend([half, diff - half])
    inputs = F.pad(inputs, pad=pad_size, mode=PytorchPadMode(padding_mode).value, value=cval)

    scan_interval = _get_scan_interval(image_size, roi_size, num_spatial_dims, overlap)

    # Store all slices in list
    slices = dense_patch_slices(image_size, roi_size, scan_interval)
    # print(f'The slices are {slices}')

    slice_batches = []
    for slice_index in range(0, len(slices), sw_batch_size):
        slice_index_range = range(slice_index, min(slice_index + sw_batch_size, len(slices)))
        input_slices = []
        for curr_index in slice_index_range:
            curr_slice = slices[curr_index]
            if len(curr_slice) == 3:
                input_slices.append(inputs[0, :, curr_slice[0], curr_slice[1], curr_slice[2]])
            else:
                input_slices.append(inputs[0, :, curr_slice[0], curr_slice[1]])
        slice_batches.append(torch.stack(input_slices))

    # Perform predictions
    output_rois = list()
    for data in slice_batches:
        if not uncertainty_flag and inputs_type == tuple:
            seg_prob, _ = predictor(data, phys_inputs)  # batched patch segmentation
            output_rois.append(seg_prob)
        elif inputs_type != tuple:
            seg_prob, _ = predictor(data)  # batched patch segmentation
            output_rois.append(seg_prob)

    # stitching output image
    output_classes = output_rois[0].shape[1]
    output_shape = [batch_size, output_classes] + list(image_size)

    # Create importance map
    importance_map = compute_importance_map(get_valid_patch_size(image_size, roi_size), mode=mode, device=inputs.device)

    # allocate memory to store the full output and the count for overlapping parts
    output_image = torch.zeros(output_shape, dtype=torch.float32, device=inputs.device)
    count_map = torch.zeros(output_shape, dtype=torch.float32, device=inputs.device)

    for window_id, slice_index in enumerate(range(0, len(slices), sw_batch_size)):
        slice_index_range = range(slice_index, min(slice_index + sw_batch_size, len(slices)))

        # store the result in the proper location of the full output. Apply weights from importance map.
        for curr_index in slice_index_range:
            curr_slice = slices[curr_index]
            if len(curr_slice) == 3:
                # print(output_image.shape, curr_slice, importance_map.shape, output_rois[window_id].shape)
                output_image[0, :, curr_slice[0], curr_slice[1], curr_slice[2]] += (
                    importance_map * output_rois[window_id][curr_index - slice_index, :]
                )
                count_map[0, :, curr_slice[0], curr_slice[1], curr_slice[2]] += importance_map
            else:
                output_image[0, :, curr_slice[0], curr_slice[1]] += (
                    importance_map * output_rois[window_id][curr_index - slice_index, :]
                )
                count_map[0, :, curr_slice[0], curr_slice[1]] += importance_map

    # account for any overlapping sections
    output_image /= count_map

    if num_spatial_dims == 3:
        return output_image[
            ...,
            pad_size[4]: image_size_[0] + pad_size[4],
            pad_size[2]: image_size_[1] + pad_size[2],
            pad_size[0]: image_size_[2] + pad_size[0],
        ]
    return output_image[
        ..., pad_size[2]: image_size_[0] + pad_size[2], pad_size[0]: image_size_[1] + pad_size[0]
    ]  # 2D


def _get_scan_interval(image_size, roi_size, num_spatial_dims: int, overlap: float):
    assert len(image_size) == num_spatial_dims, "image coord different from spatial dims."
    assert len(roi_size) == num_spatial_dims, "roi coord different from spatial dims."

    scan_interval = []
    for i in range(num_spatial_dims):
        if roi_size[i] == image_size[i]:
            scan_interval.append(int(roi_size[i]))
        else:
            # scan interval is (1-overlap)*roi_size
            scan_interval.append(int(roi_size[i] * (1 - overlap)))
    return tuple(scan_interval)


def predict_phys_seg(net, patient_data, processed_physics, main_device):
    with torch.no_grad():
        # Pass data to GPU
        if main_device == 'cpu':
            pass
        else:
            # tensor = torch.from_numpy(array)
            patient_data = torch.from_numpy(patient_data).float().cuda(main_device)
        # Basic to begin with: Just run with net!
        print(patient_data.shape, processed_physics.shape)
        if processed_physics is not None:
            out = custom_sliding_window_inference(
                (patient_data, processed_physics.cuda(main_device)), 160, 1, net, overlap=0.3, mode='gaussian')
        else:
            out = custom_sliding_window_inference(patient_data, 160, 1, net, overlap=0.3, mode='gaussian')
        pred_seg = torch.softmax(out, dim=1)
    return pred_seg.cpu().numpy()

