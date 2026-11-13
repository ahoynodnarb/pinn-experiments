import os

# import deepxde as dde
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter

# import torch

# if torch.mps.is_available():
#     torch.set_default_device("mps")
# elif torch.cuda.is_available():
#     torch.set_default_device("cuda")


a = 0.1
k = 32
N_samples = 512


def eps(theta):
    return np.random.uniform(-1e-2, 1e-2, size=len(theta))


def radius(theta):
    return 1 + a * np.sin(k * theta)


def main():
    std_real = 0.03
    l, r = -1.5, 1.5

    x_basis = np.linspace(l, r, N_samples)
    y_basis = np.linspace(l, r, N_samples)
    x, y = np.meshgrid(x_basis, y_basis)

    theta = np.atan2(y, x)
    r = np.sqrt(x**2 + y**2)
    boundary = radius(theta)
    chi = (r <= boundary).astype(float)

    # since scipy's gaussian_filter has standard deviation in pixel units, we have to transform back to coordinate units
    dx = x_basis[1] - x_basis[0]
    dy = y_basis[1] - y_basis[0]

    # Array dimensions are (y, x)
    std_pixel = (std_real / dy, std_real / dx)

    rho = gaussian_filter(
        chi,
        sigma=std_pixel,
        mode="constant",
        cval=0,
    )

    phi = rho - 0.5

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].imshow(chi, extent=[x.min(), x.max(), y.min(), y.max()], origin="lower")
    axes[0].set_title("Original indicator")

    axes[1].imshow(rho, extent=[x.min(), x.max(), y.min(), y.max()], origin="lower")
    axes[1].set_title("Gaussian convolution")

    axes[2].imshow(phi, extent=[x.min(), x.max(), y.min(), y.max()], origin="lower")
    axes[2].contour(x, y, phi, levels=[0])
    axes[2].set_title("Regularized boundary")

    for ax in axes:
        ax.set_aspect("equal")

    plt.tight_layout()
    plt.show()

    plt.show()


if __name__ == "__main__":
    main()
