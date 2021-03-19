import numpy as np
import torch


def L2norm(f):
    if f.dim() > 2:
        f_squared = torch.einsum("i...,i...->...",[f, f])
    else:
        f_squared = f * f
    return torch.sum(f_squared).sqrt()

def check_L2Diff(f1, f2): # N x size_h x size_w
    Diff = f1 - f2
    if f1.dim() > 2:
        Diff_squared = torch.einsum("i...,i...->...",[Diff, Diff])
    else:
        Diff_squared = Diff * Diff
    L2Diff = torch.sum(Diff_squared).sqrt()
    L2Error = 2*L2Diff/(L2norm(f1) + L2norm(f2))
    return L2Diff, L2Error