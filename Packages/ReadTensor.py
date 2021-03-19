import SimpleITK as sitk
import numpy as np


def readTensors(filename):
    return GetNPArrayFromSITK(sitk.ReadImage(filename), True)


def readScalar(filename):
    return GetNPArrayFromSITK(sitk.ReadImage(filename))


def GetNPArrayFromSITK(sitkimg, has_component_data=False):
    # If RGB or tensor data etc, set has_component_data to True so that last dimension is not
    # transposed.
    # This assumes that the component data is in the last dimension.
    # TODO fix this assumption to work for component data in first dimension as well
    # Currently works for 2D and 3D images
    tmp_np = sitk.GetArrayFromImage(sitkimg)
    transpose_tuple = (0, 1, 2, 3)
    # if has_component_data or (len(tmp_np.shape) != len(sitkimg.GetSize())):
    #     transpose_tuple = (1, 0, 2)
    #     if len(tmp_np.shape) == 4:
    #         print(4)
    #         transpose_tuple = (2, 1, 0, 3)
    # else:
    #     transpose_tuple = (1, 0)
    #     if len(tmp_np.shape) == 3:
    #         transpose_tuple = (2, 1, 0)

    return np.transpose(tmp_np, transpose_tuple)


if __name__ == "__main__":
    # tensor = sitk.GetArrayFromImage(sitk.ReadImage('C:/Users/Haocheng/PycharmProjects/Matching3D/Data3D/103818/dti_1000_tensor.nhdr'))
    # tensor.permute(2, 0, 1, 3)
    # tensor = GetNPArrayFromSITK(tensor, has_component_data=True)
    mask = sitk.GetArrayFromImage(
        sitk.ReadImage('C:/Users/Haocheng/PycharmProjects/Matching3D/Data3D/103818/dti_1000_FA_mask.nhdr'))
    print(mask.shape)
