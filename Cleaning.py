import torch
import SimpleITK as sitk
import scipy.io as sio

def clean(g):
    g = g.permute(2, 3, 4, 0, 1)
    for i in range(g.shape[0]):
        for j in range(g.shape[1]):
            for k in range(g.shape[2]):
                try:
                    x = torch.cholesky(g[i, j, k])
                except RuntimeError:
                    g[i, j, k] = torch.tensor([[0.001,0,0],[0,0.001,0],[0,0,0.001]],dtype=torch.double)

    return g.permute(3, 4, 0, 1, 2)


if __name__ == "__main__":
    g00 = sitk.GetArrayFromImage(sitk.ReadImage('Data3D/103818/dti_1000_tensor.nhdr'))
    g00 = torch.from_numpy(g00).double()
    g11 = sitk.GetArrayFromImage(sitk.ReadImage('Data3D/105923/dti_1000_tensor.nhdr'))
    g11 = torch.from_numpy(g11).double()
    mask = sitk.GetArrayFromImage(sitk.ReadImage('Data3D/105923/dti_1000_FA_mask.nhdr'))
    mask = torch.from_numpy(mask).double().unsqueeze(0)

    height, width, depth = g00.shape[-3:]
    g0, g1 = torch.zeros(3, 3, height, width, depth, dtype=torch.double), torch.zeros(3, 3, height, width, depth,
                                                                                      dtype=torch.double)
    g0[0, 0] = g00[0]
    g0[0, 1] = g00[1]
    g0[0, 2] = g00[2]
    g0[1, 0] = g00[1]
    g0[1, 1] = g00[3]
    g0[1, 2] = g00[4]
    g0[2, 0] = g00[2]
    g0[2, 1] = g00[4]
    g0[2, 2] = g00[5]

    g1[0, 0] = g11[0]
    g1[0, 1] = g11[1]
    g1[0, 2] = g11[2]
    g1[1, 0] = g11[1]
    g1[1, 1] = g11[3]
    g1[1, 2] = g11[4]
    g1[2, 0] = g11[2]
    g1[2, 1] = g11[4]
    g1[2, 2] = g11[5]

    g1 = clean(g1)
    g0 = clean(g0)

    g00[0] = g0[0, 0]
    g00[1] = g0[0, 1]
    g00[2] = g0[0, 2]
    g00[3] = g0[1, 1]
    g00[4] = g0[1, 2]
    g00[5] = g0[2, 2]

    g11[0] = g1[0, 0]
    g11[1] = g1[0, 1]
    g11[2] = g1[0, 2]
    g11[3] = g1[1, 1]
    g11[4] = g1[1, 2]
    g11[5] = g1[2, 2]

    sio.savemat('103818_dti_1000_tensor_cleaned.mat', {'tensor': g00.detach().numpy()})
    sio.savemat('105923_dti_1000_tensor_cleaned.mat', {'tensor': g11.detach().numpy()})