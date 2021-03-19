import numpy as np
import torch
from scipy.linalg import expm, logm
import warnings
from Packages.RegistrationFunc import *
from Packages.SplitEbinMetric import *
from Packages.GeoPlot import *
import SimpleITK as sitk

if __name__ == "__main__":
    torch.set_default_tensor_type('torch.cuda.DoubleTensor')
    A = torch.rand(145, 174, 145, 3, 3) + 0.001
    A[:, :, :, 0, 1] = 0
    A[:, :, :, 0, 2] = 0
    A[:, :, :, 1, 2] = 0
    A = torch.einsum("...ij,...kj->...ik", A, A)

    B = torch.rand(145, 174, 145, 3, 3) + 0.001
    B[:, :, :, 0, 1] = 0
    B[:, :, :, 0, 2] = 0
    B[:, :, :, 1, 2] = 0
    B = torch.einsum("...ij,...kj->...ik", B, B)

    G = torch.stack((A, B)).to(device=torch.device('cuda'))
    gm = get_KarcherMean(G, 1. / 3.)

    plot_2d_tensors(A[:, :, 87, :2, :2])
    plot_2d_tensors(B[:, :, 87, :2, :2])
    plot_2d_tensors(gm[:, :, 87, :2, :2])
