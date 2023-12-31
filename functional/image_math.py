import torch
import numpy as np
import torch.nn.functional as F


def numpy_gaussian_kernel(ksize=3, ksigma=2):
    kernel = np.fromfunction(
        lambda x, y: (1 / (2 * np.pi * ksigma**2))
        * np.exp(
            -((x - (ksize // 2)) ** 2 + (y - (ksize // 2)) ** 2) / (2 * ksigma**2)
        ),
        (ksize, ksize),
    )
    return kernel / np.sum(kernel)


def tensor_gaussian_kernel(ksize=3, ksigma=2):
    x, y = torch.meshgrid(torch.arange(ksize), torch.arange(ksize), indexing="ij")
    x_centered = x - ksize // 2
    y_centered = y - ksize // 2

    kernel = (1 / (2 * torch.pi * ksigma**2)) * torch.exp(
        -(x_centered**2 + y_centered**2) / (2 * ksigma**2)
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
