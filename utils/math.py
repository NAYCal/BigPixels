import torch
import numpy as np

def numpy_gaussian_kernel(size=3, sigma= (3 - 1)//6):
    kernel = np.fromfunction(
        lambda x, y: (1 / (2 * np.pi * sigma ** 2))
                     * np.exp(-((x - (size // 2)) ** 2 + (y - (size // 2)) ** 2) / (2 * sigma ** 2)),
        (size, size),
    )
    return kernel / np.sum(kernel)

def tensor_gaussian_kernel(size=3, sigma= (3 - 1)//6):
    x, y = torch.meshgrid(torch.arange(size), torch.arange(size), indexing='ij')
    x_centered = x - size // 2
    y_centered = y - size // 2

    kernel = (1 / (2 * torch.pi * sigma ** 2)) * torch.exp(-(x_centered ** 2 + y_centered ** 2) / (2 * sigma ** 2))
    kernel = kernel / torch.sum(kernel)

    return kernel
