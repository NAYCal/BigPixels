import torch
import numpy as np
import torch.nn.functional as F


def numpy_gaussian_kernel(size=3, sigma=(3 - 1) / 6):
    kernel = np.fromfunction(
        lambda x, y: (1 / (2 * np.pi * sigma**2))
        * np.exp(-((x - (size // 2)) ** 2 + (y - (size // 2)) ** 2) / (2 * sigma**2)),
        (size, size),
    )
    return kernel / np.sum(kernel)


def tensor_gaussian_kernel(size=3, sigma=(3 - 1) / 6):
    x, y = torch.meshgrid(torch.arange(size), torch.arange(size), indexing="ij")
    x_centered = x - size // 2
    y_centered = y - size // 2

    kernel = (1 / (2 * torch.pi * sigma**2)) * torch.exp(
        -(x_centered**2 + y_centered**2) / (2 * sigma**2)
    )
    kernel = kernel / torch.sum(kernel)

    return kernel

def naive_ssd(arr1, arr2):
    return ((arr1 - arr2) ** 2).sum()


def ssd(arr1, arr2):
    assert type(arr1) == type(arr2), "Both arrays have to be of the same type!"
    if isinstance(arr1, torch.Tensor):
        return torch_ssd(arr1, arr2)
    elif isinstance(arr1, np.ndarray):
        return numpy_ssd(arr1, arr2)


def numpy_ssd(arr1, arr2):
    diff = arr1.ravel() - arr2.ravel()
    return np.dot(diff, diff)


def torch_ssd(arr1, arr2):
    first_term = torch.sum(torch.square(arr1)).item()
    second_term = F.conv2d(
        arr1.unsqueeze(0), arr2.unsqueeze(0), padding=0, stride=1
    ).item()
    third_term = torch.sum(torch.square(arr2)).item()

    return first_term - 2 * second_term + third_term
