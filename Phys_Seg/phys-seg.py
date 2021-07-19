import os
from Phys_Seg.run import run_phys_seg
from Phys_Seg.utils import maybe_mkdir_p, subfiles
import Phys_Seg

if __name__ == "__main__":
    print("\n########################")
    print("########################\n")

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', help='input. Can be either a single file name or an input folder. If file: must be '
                                       'nifti (.nii.gz) and can only be 3D. No support for 4d images, use fslsplit to '
                                       'split 4d sequences into 3d images. If folder: all files ending with .nii.gz '
                                       'within that folder will be brain extracted.', required=True, type=str)
    parser.add_argument('-o', '--output', help='output. Can be either a filename or a folder. If it does not exist, the folder'
                                               ' will be created', required=False, type=str)
    parser.add_argument('--sequence', type=str, default='MPRAGE', help='',
                        required=True)
    parser.add_argument('--physics_params', type=str, default=None, help='Physics parameters.'
                                                                         'For MPRAGE, specify with square brackets [TI, TR]',
                        required=False)
    parser.add_argument('--device', default='0', type=str, help='used to set on which device the prediction will run. '
                                                                'Must be either int or str. Use int for GPU id or '
                                                                '\'cpu\' to run on CPU. When using CPU you should '
                                                                'consider disabling tta. Default for -device is: 0',
                        required=False)
    parser.add_argument('--overwrite_existing', default=1, type=int, required=False, help="set this to 0 if you don't "
                                                                                          "want to overwrite existing "
                                                                                          "predictions")

    args = parser.parse_args()

    input_file_or_dir = args.input
    output_file_or_dir = args.output

    if output_file_or_dir is None:
        output_file_or_dir = os.path.join(os.path.dirname(input_file_or_dir),
                                          os.path.basename(input_file_or_dir).split(".")[0] + "_phys_seg")

    sequence = args.sequence
    if sequence is not None:
        assert type(eval(sequence)) == list, 'Physics parameters should be specified between square brackets!'
    print(sequence)
    physics_params = args.physics_params
    device = args.device
    overwrite_existing = args.overwrite_existing

    # params_file = os.path.join(PHYS_SEG.__path__[0], "model_final.py")
    # config_file = os.path.join(PHYS_SEG.__path__[0], "config.py")

    assert os.path.abspath(input_file_or_dir) != os.path.abspath(output_file_or_dir), "output must be different from input"

    if device == 'cpu':
        pass
    else:
        device = int(device)

    if os.path.isdir(input_file_or_dir):
        maybe_mkdir_p(output_file_or_dir)
        input_files = subfiles(input_file_or_dir, suffix='.nii.gz', join=False)

        if len(input_files) == 0:
            raise RuntimeError("input is a folder but no nifti files (.nii.gz) were found in here")

        output_files = [os.path.join(output_file_or_dir, i) for i in input_files]
        input_files = [os.path.join(input_file_or_dir, i) for i in input_files]
    else:
        if not output_file_or_dir.endswith('.nii.gz'):
            output_file_or_dir += '.nii.gz'
            assert os.path.abspath(input_file_or_dir) != os.path.abspath(output_file_or_dir), "output must be different from input"

        output_files = [output_file_or_dir]
        input_files = [input_file_or_dir]

    if overwrite_existing == 0:
        overwrite_existing = False
    elif overwrite_existing == 1:
        overwrite_existing = True
    else:
        raise ValueError("Unknown value for overwrite_existing: %s. Expected: 0 or 1" % str(overwrite_existing))

    run_phys_seg(input_files, output_files, sequence, physics_params, device, overwrite_existing)
