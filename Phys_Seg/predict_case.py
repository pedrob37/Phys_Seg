import torch
import numpy as np


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
    physics_input = physics_input[None, :]
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
    return overall_physics


def predict_phys_seg(net, patient_data, processed_physics, main_device=0):
    with torch.no_grad():
        # Pass data to GPU
        if main_device == 'cpu':
            pass
        else:
            # tensor = torch.from_numpy(array)
            patient_data = torch.from_numpy(patient_data).cuda(main_device)
        # Basic to begin with: Just run with net!
        if processed_physics is not None:
            out, _ = net(patient_data, torch.from_numpy(processed_physics).cuda(main_device))
        else:
            out, _ = net(patient_data)
        pred_seg = torch.softmax(out, dim=1)
    return pred_seg
