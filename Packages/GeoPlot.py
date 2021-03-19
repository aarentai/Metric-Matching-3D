import matplotlib.pyplot as plt
import torch
import numpy as np
from matplotlib.patches import Ellipse
from matplotlib.collections import PatchCollection


def plot_2d_tensors(nda0, scale=0.5, title=None, margin=0.05, dpi=80):
    nda = torch.zeros(*nda0.shape[:2], 3)
    nda[:, :, 0] = nda0[:, :, 0, 0]
    nda[:, :, 1] = nda0[:, :, 0, 1]
    nda[:, :, 2] = nda0[:, :, 1, 1]
    nda = nda.cpu().numpy()

    if nda.ndim == 3:
        # fastest dim, either component or x
        c = nda.shape[-1]
        # the number of component is 3 consider it a tensor image
        if c != 3:
            raise Exception("Unable to show 3D-vector Image")

    xsize = nda.shape[0]
    ysize = nda.shape[1]

    # Make a figure big enough to accommodate an axis of xpixels by ypixels
    # as well as the ticklabels, etc...
    if xsize > dpi and ysize > dpi:
        figsize = (1 + margin) * ysize / dpi, (1 + margin) * xsize / dpi
    else:
        figsize = (1 + margin) * dpi / ysize, (1 + margin) * dpi / xsize
    # print("fig size: ", figsize)

    # fig = plt.figure(figsize=figsize, dpi=dpi)

    plt.ion()
    plt.show()
    fig = plt.figure()
    # Make the axis the right size...
    ax = fig.add_axes([margin, margin, 1 - 2 * margin, 1 - 2 * margin])

    #     extent = (0, xsize*spacing[1], ysize*spacing[0], 0)
    tens = np.zeros((2, 2))
    triu_idx = np.triu_indices(2)
    ellipses = []
    xax = [1, 0]
    for x in range(xsize):
        for y in range(ysize):
            tens[triu_idx] = nda[x, y]
            tens[1, 0] = tens[0, 1]
            # evals, evecs = np.linalg.eigh(nda[x,y])
            evals, evecs = np.linalg.eigh(tens)
            # angle = np.degrees(np.math.atan2(np.linalg.det([xax,evecs[1]]),np.dot(xax,evecs[1])))
            angle = np.degrees(np.math.atan2(evecs[1][1], evecs[1][0]))
            ellipses.append(Ellipse((x, y), width=scale * evals[1], height=scale * evals[0], angle=angle))
    collection = PatchCollection(ellipses, alpha=0.7)
    ax.add_collection(collection)
    ax.set_xlim(0, xsize)
    ax.set_ylim(0, ysize)
    ax.set_aspect('equal')

    if (title):
        plt.title(title)

    plt.draw()
    plt.pause(0.001)


def plot_PDSMs(e, v):  # e:(s,t,2), v:(s,t,2,2)
    t = torch.linspace(0, 2 * np.pi, 100, dtype=torch.double)
    Tri = torch.stack((torch.cos(t), torch.sin(t)), axis=0)
    P = torch.einsum("...ik,...k,kj->...ij", [v, e.sqrt(), Tri])  # (s,t,2,100)
    if e.dim() == 1:
        plt.plot(P[0].numpy(), P[1].numpy(), '-k', lw=1)
    if e.dim() == 2:
        for i in range(e.size(0)):
            Px = P[i, 0] + i
            Py = P[i, 1]
            plt.plot(Px.numpy(), Py.numpy(), '-k', lw=1)
    if e.dim() == 3:
        for i in range(e.size(0)):
            for j in range(e.size(1)):
                Px = P[i, j, 0] + i
                Py = P[i, j, 1] + j
                plt.plot(Px.numpy(), Py.numpy(), '-k', lw=1)


def plot_geo(geo, num_figs):  # (l,...,2,2)
    e, v = torch.symeig(geo, eigenvectors=True)
    fig = plt.figure(figsize=(30, 30))
    for i in range(num_figs):  # plot num_figs figures
        ax = fig.add_subplot(1, num_figs, i + 1)
        ax.set_aspect('equal')
        i = round((geo.size(0) - 1) * i / (num_figs - 1))
        plot_PDSMs(e[i] / (e.max()), v[i])  # /(2*e.max())
        #         plt.axis('off')
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        plt.axis('scaled')
    #         ax.set_xlim(-1.1, 1.1)
    #         ax.set_ylim(-1, 1)
    #         plt.gca().set_aspect('equal', adjustable='box')
    plt.show()


def plot_diffeo(diffeo, title=None, step_size=1, show_axis=False):
    diffeo = diffeo.cpu().detach().numpy()
    #     diffeo = diffeo.numpy()
    plt.ion()
    plt.show()
    plt.figure(num=None, figsize=(5, 5), dpi=100, facecolor='w', edgecolor='k')
    if show_axis is False:
        plt.axis('off')
    ax = plt.gca()
    ax.set_aspect('equal')
    #     ax.invert_yaxis()
    #     plt.plot(diffeo[1,:,:], diffeo[0,:,:],'b')
    #     plt.plot((diffeo[1,:,:]).t(), (diffeo[0,:,:]).t(),'b')
    for h in range(0, diffeo.shape[1], step_size):
        # plt.plot(diffeo[1, h, :], diffeo[0, h, :], 'b', linewidth=0.5)
        plt.plot(diffeo[0, h, :], diffeo[1, h, :], 'b', linewidth=0.5)
    for w in range(0, diffeo.shape[2], step_size):
        # plt.plot(diffeo[1, :, w], diffeo[0, :, w], 'b', linewidth=0.5)
        plt.plot(diffeo[0, :, w], diffeo[1, :, w], 'b', linewidth=0.5)

    if (title):
        plt.title(title)

    plt.draw()
    plt.pause(0.001)


def plot_img(I):
    fig = plt.figure(num=None, figsize=(6, 3), dpi=100, facecolor='w', edgecolor='k')
    img = plt.imshow(I)
    plt.colorbar(img, fraction=0.046, pad=0.04)


def plot_energy(E):
    # plt.ion()
    # plt.show()
    plt.figure(num=None, figsize=(5, 5), dpi=100, facecolor='w', edgecolor='k')
    plt.plot(E, label="Energy")
    # plt.draw()
    # plt.pause(0.001)
