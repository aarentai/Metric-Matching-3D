import SimpleITK as sitk
import torch
from Packages.SplitEbinMetric3D import *


if __name__ == "__main__":
    # brain_id = ['103818', '105923']
    # g_orig = sitk.GetArrayFromImage(sitk.ReadImage('Data3D/103818/dti_1000_tensor.nhdr'))
    # g_orig = torch.from_numpy(g_orig).double()  # [6, 145, 174, 145]
    # height = g_orig.size(-3)  # 145
    # width = g_orig.size(-2)  # 174
    # depth = g_orig.size(-1)  # 145
    # G = torch.zeros(len(brain_id), height, width, depth, 3, 3, dtype=torch.double)  # [2, 145, 174, 145, 3, 3]
    #
    # for i in range(len(brain_id)):
    #     g_orig = sitk.GetArrayFromImage(sitk.ReadImage('Data3D/' + brain_id[i] + '/dti_1000_tensor.nhdr'))
    #     g_orig = torch.from_numpy(g_orig).double()  # [6, 145, 174, 145]
    #     g_met = torch.zeros(height, width, depth, 3, 3, dtype=torch.double)
    #     g_met[:, :, :, 0, 0] = g_orig[0]
    #     g_met[:, :, :, 0, 1] = g_orig[1]
    #     g_met[:, :, :, 0, 2] = g_orig[2]
    #     g_met[:, :, :, 1, 0] = g_orig[1]
    #     g_met[:, :, :, 1, 1] = g_orig[3]
    #     g_met[:, :, :, 1, 2] = g_orig[4]
    #     g_met[:, :, :, 2, 0] = g_orig[2]
    #     g_met[:, :, :, 2, 1] = g_orig[4]
    #     g_met[:, :, :, 2, 2] = g_orig[5]
    #     G[i] = g_met  # [2, 145, 174, 145, 3, 3]


    A = torch.rand(145, 174, 145, 3, 3, dtype=torch.double)
    A = torch.einsum("...ij,...kj->...ik", A, A)

    B = torch.rand(145, 174, 145, 3, 3, dtype=torch.double)
    B = torch.einsum("...ij,...kj->...ik", B, B)
    G = torch.stack((A, B))
    a = 1. / 3.
    gm = get_KarcherMean(G, a)
    print(torch.sum(gm))
    # my = logm_invB_A(B,A)
    # kl = logm_invB_A_2d(B,A)
    # print(torch.norm(my-kl))