import numpy as np
import torch
from Packages.RegistrationFunc import *
from Packages.SplitEbinMetric import *
from Packages.GeoPlot import *
import scipy.io as sio
import os
import matplotlib.pyplot as plt
import SimpleITK as sitk
from scipy.io import savemat


# define the pullback action of phi
def phi_pullback(phi, g):
#     input: phi.shape = [3, h, w, d]; g.shape = [h, w, d, 3, 3]
#     output: shape = [h, w, 3, 3]
#     torch.set_default_tensor_type('torch.cuda.DoubleTensor')
    g = g.permute(3, 4, 0, 1, 2)
    idty = get_idty(*g.shape[-3:])
    #     four layers of scalar field, of all 1, all 0, all 1, all 0, where the shape of each layer is g.shape[-2:]?
    eye = torch.eye(3)
    ones = torch.ones(*g.shape[-3:])
    d_phi = get_jacobian_matrix(phi - idty) + torch.einsum("ij,mno->ijmno", eye, ones)
    g_phi = compose_function(g, phi)
    return torch.einsum("ij...,ik...,kl...->...jl", d_phi, g_phi, d_phi)


# define the energy functional
def energy_ebin(phi, g0, g1, f0, f1, sigma, lambd, mask):
#     input: phi.shape = [3, h, w, d]; g0/g1/f0/f1.shape = [h, w, d, 3, 3]; sigma/lambd = scalar; mask.shape = [1, h, w, d]
#     output: scalar
# the phi here is identity
    phi_star_g1 = phi_pullback(phi, g1)
    phi_star_f1 = phi_pullback(phi, f1)# the compose operation in this step uses a couple of thousands MB of memory
    E1 = sigma * Squared_distance_Ebin(f0, phi_star_f1, lambd, mask)
    E2 = Squared_distance_Ebin(g0, phi_star_g1, lambd, mask)
    return E1 + E2


# define the energy functional
def energy_l2(phi, g0, g1, f0, f1, sigma, mask):
#     input: phi.shape = [3, h, w, d]; g0/g1/f0/f1.shape = [h, w, d, 3, 3]; sigma/lambd = scalar; mask.shape = [1, h, w, d]
#     output: scalar
    phi_star_g1 = phi_pullback(phi, g1)
    phi_star_f1 = phi_pullback(phi, f1)
    E1 = sigma * torch.einsum("ijk...,lijk->", (f0 - phi_star_f1) ** 2, mask)
    E2 = torch.einsum("ijk...,lijk->", (g0 - phi_star_g1) ** 2, mask)
    # E = E1 + E2
    # del phi_star_g1, phi_star_f1
    # torch.cuda.empty_cache()
    return E1 + E2


def laplace_inv(u):
    '''
    this function computes the laplacian inverse of a vector field u of size 3 x size_h x size_w x size_d
    '''
    size_h, size_w, size_d = u.shape[-3:]
    idty = get_idty(size_h, size_w, size_d).cpu().numpy()
    lap = 6. - 2. * (np.cos(2. * np.pi * idty[0] / size_h) +
                     np.cos(2. * np.pi * idty[1] / size_w) +
                     np.cos(2. * np.pi * idty[2] / size_d))
    lap[0, 0] = 1.
    lapinv = 1. / lap
    lap[0, 0] = 0.
    lapinv[0, 0] = 1.

    u = u.cpu().detach().numpy()
    fx = np.fft.fftn(u[0])
    fy = np.fft.fftn(u[1])
    fz = np.fft.fftn(u[2])
    fx *= lapinv
    fy *= lapinv
    fz *= lapinv
    vx = torch.from_numpy(np.real(np.fft.ifftn(fx)))
    vy = torch.from_numpy(np.real(np.fft.ifftn(fy)))
    vz = torch.from_numpy(np.real(np.fft.ifftn(fz)))

    return torch.stack((vx, vy, vz))#.to(device=torch.device('cuda'))


def checkNaN(A):
    if (A != A).any():
        print('NaN')


if __name__ == "__main__":
    # torch.cuda.empty_cache()
    # torch.set_default_tensor_type('torch.cuda.DoubleTensor')
    torch.set_default_tensor_type('torch.DoubleTensor')
    # g00 = sitk.GetArrayFromImage(sitk.ReadImage('Data3D/103818/dti_1000_tensor.nhdr'))
    g00 = sitk.GetArrayFromImage(sitk.ReadImage('/usr/sci/projects/HCP/Kris/NSFCRCNS/TestData/3D/metpy_3D_cubic4_tens.nhdr'))
    g00 = torch.from_numpy(g00).double().permute(3,2,1,0)
    # g11 = sitk.GetArrayFromImage(sitk.ReadImage('Data3D/105923/dti_1000_tensor.nhdr'))
    g11 = sitk.GetArrayFromImage(sitk.ReadImage('/usr/sci/projects/HCP/Kris/NSFCRCNS/TestData/3D/metpy_3D_cubic6_tens.nhdr'))
    g11 = torch.from_numpy(g11).double().permute(3,2,1,0)
    # mask = sitk.GetArrayFromImage(sitk.ReadImage('Data3D/105923/dti_1000_FA_mask.nhdr'))
    mask = sitk.GetArrayFromImage(sitk.ReadImage('/usr/sci/projects/HCP/Kris/NSFCRCNS/TestData/3D/metpy_3D_cubic6_mask.nhdr'))
    mask = torch.from_numpy(mask).double().permute(2,1,0).unsqueeze(0)#.to(device='cuda') #.half()

    height, width, depth = g00.shape[-3:]
    g0 = torch.zeros(height, width, depth, 3, 3)
    g1 = torch.zeros(height, width, depth, 3, 3)
    g0[:,:,:,0, 0] = g00[0]
    g0[:,:,:,0, 1] = g00[1]
    g0[:,:,:,0, 2] = g00[2]
    g0[:,:,:,1, 0] = g00[1]
    g0[:,:,:,1, 1] = g00[3]
    g0[:,:,:,1, 2] = g00[4]
    g0[:,:,:,2, 0] = g00[2]
    g0[:,:,:,2, 1] = g00[4]
    g0[:,:,:,2, 2] = g00[5]

    g1[:,:,:,0, 0] = g11[0]
    g1[:,:,:,0, 1] = g11[1]
    g1[:,:,:,0, 2] = g11[2]
    g1[:,:,:,1, 0] = g11[1]
    g1[:,:,:,1, 1] = g11[3]
    g1[:,:,:,1, 2] = g11[4]
    g1[:,:,:,2, 0] = g11[2]
    g1[:,:,:,2, 1] = g11[4]
    g1[:,:,:,2, 2] = g11[5]
    # del g00, g11
    # torch.cuda.empty_cache()

    lambd = 1. / 3.
    sigma = 0
    # epsilon = 5e4 # without ada
    epsilon = 5
    Num_ite = 20#20

    phi_inv = get_idty(height, width, depth)
    phi = get_idty(height, width, depth)

    f0 = torch.eye(3).repeat(height, width, depth, 1, 1)
    f1 = torch.eye(3).repeat(height, width, depth, 1, 1)

    idty = get_idty(height, width, depth)
    idty.requires_grad_()
    E = torch.tensor([np.inf])

    grad1_all = []
    E_all = []

    # L2
    # for i in range(Num_ite):
    #     phi_actsg0 = phi_pullback(phi_inv, g0)
    #     phi_actsf0 = phi_pullback(phi_inv, f0)

    #     E0 = E.clone()
    #     E = energy_l2(idty, phi_actsg0, g1, phi_actsf0, f1, sigma, mask)  # , lambd)
    #     E.backward()

    #     if (E - E0) / E0 > 1e-8:
    #         print(i)
    #         epsilon = epsilon / 1.5

    #     v = - laplace_inv(idty.grad)
    #     # alpha = epsilon
    #     alpha = epsilon / torch.norm(idty.grad)
    #     E_all.append(E.item())

    #     with torch.no_grad():
    #         psi = idty + alpha * v
    #         psi[0][psi[0] > height - 1] = height - 1
    #         psi[1][psi[1] > width - 1] = width - 1
    #         psi[2][psi[2] > depth - 1] = depth - 1
    #         psi[psi < 0] = 0

    #         psi_inv = idty - alpha * v
    #         psi_inv[0][psi_inv[0] > height - 1] = height - 1
    #         psi_inv[1][psi_inv[1] > width - 1] = width - 1
    #         psi_inv[2][psi_inv[2] > depth - 1] = depth - 1
    #         psi_inv[psi_inv < 0] = 0

    #         phi = compose_function(psi, phi)
    #         phi_inv = compose_function(phi_inv, psi_inv)

    #         # if i % 100 == 0:
    #         diffeo_slice = torch.stack((phi[1, 70, :, :], phi[2, 70, :, :]))
    #         # diffeo_slice = torch.stack((phi[0, :, 70, :], phi[2, :, 70, :]))
    #         # diffeo_slice = torch.stack((phi[0, :, :, 70], phi[1, :, :, 70]))
    #         plot_diffeo(diffeo_slice, step_size=2, show_axis=True)
    #         print(E.item())

    #         idty.grad.data.zero_()

    #     if abs((E - E0) / E0) < 1e-7:
    #         break

    plt.plot(E_all)
    g11_n = phi_pullback(phi_inv, g0)

    alpha = 1e-2
    Num_ite = 20#100
    count = 0

    #
    # f0 = torch.eye(3, dtype=torch.double).repeat(height, width, depth, 1, 1).permute(3, 4, 0, 1, 2)
    # f1 = torch.eye(3, dtype=torch.double).repeat(height, width, depth, 1, 1).permute(3, 4, 0, 1, 2)
    # E = torch.tensor([np.inf], dtype=torch.double)
    #
    for i in range(Num_ite):
        print(i)
        phi_actsg0 = phi_pullback(phi_inv, g0)
        phi_actsf0 = phi_pullback(phi_inv, f0)

        # with torch.autograd.set_detect_anomaly(True):
        E0 = E.clone()
        E = energy_ebin(idty, phi_actsg0, g1, phi_actsf0, f1, sigma, lambd, mask)
        E.backward()

        # if E > E0 and (E - E0) / E0 > 0:
        #     alpha = alpha/2
        #     count+=1
        #     print(count)

        v = - laplace_inv(idty.grad)
        # alpha = epsilon
        alpha = epsilon / torch.norm(idty.grad)
        E_all.append(E.item())

        with torch.no_grad():
            psi = idty + alpha * v
            psi[0][psi[0] > height - 1] = height - 1
            psi[1][psi[1] > width - 1] = width - 1
            psi[2][psi[2] > depth - 1] = depth - 1
            psi[psi < 0] = 0

            psi_inv = idty - alpha * v
            psi_inv[0][psi_inv[0] > height - 1] = height - 1
            psi_inv[1][psi_inv[1] > width - 1] = width - 1
            psi_inv[2][psi_inv[2] > depth - 1] = depth - 1
            psi_inv[psi_inv < 0] = 0

            phi = compose_function(psi, phi)
            phi_inv = compose_function(phi_inv, psi_inv)

            plot_diffeo(phi[:3, :, :, 20], step_size=2, show_axis=True)
            print(E.item())

            idty.grad.data.zero_()

        if abs((E - E0) / E0) < 1e-11 or count >= 200:
            break
    #
    # print(E_all[-1])
    # plot_energy(E_all)
    # plot_diffeo(phi, title="phi", step_size=2, show_axis=True)
    # plot_diffeo(phi_inv, title="phi_inv", step_size=2, show_axis=True)
    #
    # sio.savemat("s_phi.mat", {'a': phi.numpy(), 'label': 'phi'})
    # sio.savemat("s_phi_inv.mat", {'a': phi_inv.numpy(), 'label': 'phi'})
    #
    # g1_n = phi_pullback(phi_inv, g0)
    # det_phi = get_jacobian_determinant(phi)
    # plt.figure()
    # plt.imshow(det_phi[1:99, 1:99])
    # os.system("pause")

    # plot_2d_tensors(g1, scale=0.5, title="g1", margin=0.05, dpi=80)
    # plot_2d_tensors(g11_n, scale=0.5, title="g11_n", margin=0.05, dpi=80)
    # plot_2d_tensors(g1_n, scale=0.5, title="g1_n", margin=0.05, dpi=80)
    # plot_2d_tensors(g0, scale=0.5, title="g0", margin=0.05, dpi=80)
