import torch
import numpy as np
import SimpleITK as sitk
from Phys_Seg.data_loading import load_and_preprocess, save_segmentation_nifti
from Phys_Seg.predict_case import predict_phys_seg, physics_preprocessing, image_preprocessing
import importlib
from Phys_Seg.utils import postprocess_prediction, get_params_fname, maybe_download_parameters
from network_architecture import nnUNet
import os
import Phys_Seg


def apply_phys_seg(img, out_fname):
    img_itk = sitk.ReadImage(img)
    img_npy = sitk.GetArrayFromImage(img_itk)
    out = sitk.GetImageFromArray(img_npy)
    out.CopyInformation(img_itk)
    sitk.WriteImage(out, out_fname)


def run_phys_seg(mri_fnames, output_fnames, sequence='MPRAGE', physics_params=None,
                 # config_file=os.path.join(Phys_Seg.__path__[0], "config.py"),
                 device=0, overwrite=True):
    """

    :param mri_fnames: str or list/tuple of str
    :param output_fnames: str or list/tuple of str. If list: must have the same length as output_fnames
    :param sequence: MPRAGE or SPGR (for now)
    :param config_file: config.py
    :param device: either int (for device id) or 'cpu'
    :param postprocess: whether to do postprocessing or not. Postprocessing here consists of simply discarding all
    but the largest predicted connected component. Default False
    :return:
    """

    physics_input_size = {'MPRAGE': 4,
                          'SPGR': 6}

    # Load in model weights
    maybe_download_parameters(sequence=sequence, physics_flag=True if physics_params else False)
    params_file = get_params_fname(sequence=sequence, physics_flag=True if physics_params else False)

    net = nnUNet(4, 4, physics_flag=True if physics_params else False,
                 physics_input=physics_input_size[sequence],
                 physics_output=40)

    if device == "cpu":
        net = net.cpu()
    else:
        net.cuda(device)

    if not isinstance(mri_fnames, (list, tuple)):
        mri_fnames = [mri_fnames]

    if not isinstance(output_fnames, (list, tuple)):
        output_fnames = [output_fnames]

    params = torch.load(params_file, map_location=lambda storage, loc: storage)

    for in_fname, out_fname in zip(mri_fnames, output_fnames):
        if overwrite or not (os.path.isfile(out_fname)):
            print("File:", in_fname)
            print("preprocessing...")
            try:
                data, data_dict = load_and_preprocess(in_fname)
            except RuntimeError:
                print("\nERROR\nCould not read file", in_fname, "\n")
                continue
            except AssertionError as e:
                print(e)
                continue

            # Process data
            if physics_params is not None:
                processed_physics = physics_preprocessing(physics_params, sequence)
            data = image_preprocessing(patient_data=data)

            print("prediction (CNN id)...")
            net.load_state_dict(params['model_state_dict'])
            net.eval()
            seg = predict_phys_seg(net=net,
                                   patient_data=data,
                                   processed_physics=processed_physics,
                                   main_device=device)

            # if postprocess:
            #     seg = postprocess_prediction(seg)

            print("exporting segmentation...")
            save_segmentation_nifti(seg, data_dict, out_fname)

            apply_phys_seg(in_fname, out_fname)
