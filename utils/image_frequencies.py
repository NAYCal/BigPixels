import torch
import numpy as np

from scipy.signal import convolve2d
from torch.nn.functional import conv2d
from image_io import normalize_image
from image_math import numpy_gaussian_kernel, tensor_gaussian_kernel


def finite_difference(
    image,
    reduce_noise=True,
    out_grayscale=True,
    threshold=None,
    ksize=30,
    ksigma=30,
    device="cpu",
):
    match type(image):
        case np.ndarray:
            return numpy_finite_difference(
                image,
                reduce_noise=reduce_noise,
                out_grayscale=out_grayscale,
                ksize=ksize,
                ksigma=ksigma,
                threshold=threshold,
            )
        case torch.Tensor:
            return tensor_finite_difference(
                image,
                reduce_noise=reduce_noise,
                out_grayscale=out_grayscale,
                ksize=ksize,
                ksigma=ksigma,
                threshold=threshold,
                device=device,
            )
        case _:
            raise TypeError("Image type not supported!")


def numpy_finite_difference(
    image, reduce_noise=True, out_grayscale=True, ksize=30, ksigma=30, threshold=None
):
    cov = lambda img, kernel: convolve2d(
        img, kernel, mode="same", boundary="fill", fillvalue=0
    )

    dx = np.array([[1, 0, -1]])
    dy = dx.T
    if reduce_noise:
        gaussian_kernel = numpy_gaussian_kernel(ksize=ksize, ksigma=ksigma)
        dx = cov(dx, gaussian_kernel)
        dy = cov(dy, gaussian_kernel)

    match len(image.shape):
        case 2:
            image_dx = cov(image, dx)
            image_dy = cov(image, dy)
            derived = np.sqrt((image_dx**2) + (image_dy**2))
            return normalize_image(
                (derived > threshold).astype(np.float32) if threshold else derived
            )
        case 3:
            channels = []
            for d in range(3):
                channel_dx = cov(image[:, :, d], dx)
                channel_dy = cov(image[:, :, d], dy)
                channels.append(np.sqrt((channel_dx**2) + (channel_dy**2)))

            derived = np.stack(channels, axis=-1)
            derived = np.mean(derived, axis=2) if out_grayscale else derived
            return normalize_image(
                (derived > threshold).astype(np.float32) if threshold else derived
            )
        case 4:
            diff_images = np.stack(
                [
                    numpy_finite_difference(
                        image[i], reduce_noise, out_grayscale, threshold
                    )
                    for i in image.shape[0]
                ],
                axis=0,
            )
            return diff_images
        case _:
            raise ValueError("Image must be a 2D object!")


def tensor_finite_difference(
    image,
    reduce_noise=True,
    out_grayscale=True,
    ksize=3,
    ksigma=2,
    threshold=None,
    device="cpu",
):
    original_shape = image.shape
    image = image.to(device)
    dx = torch.tensor([[[[-1, 0, 1]]]], dtype=torch.float, device=device)
    dy = torch.tensor([[[[-1], [0], [1]]]], dtype=torch.float, device=device)
    if reduce_noise:
        gaussian_kernel = (
            tensor_gaussian_kernel(ksize=ksize, ksigma=ksigma)
            .unsqueeze(0)
            .unsqueeze(0)
            .to(device=device)
        )
        dx = conv2d(dx, gaussian_kernel, padding="same")
        dy = conv2d(dy, gaussian_kernel, padding="same")

    channel_dim = None
    match len(image.shape):
        case 2:
            image = image.unsqueeze(0).unsqueeze(0)
        case 3:
            channel_dim = 0
            height, width = image.shape[1:]
            image = image.reshape(-1, 1, height, width)
        case 4:
            channel_dim = 1
        case _:
            raise ValueError("Image must be a 2D object!")

    image_dx = conv2d(image, dx, padding="same")
    image_dy = conv2d(image, dy, padding="same")
    derived = torch.sqrt(image_dx**2 + image_dy**2)
    derived = derived.view(original_shape)

    if channel_dim is not None and out_grayscale:
        derived = torch.mean(derived, dim=channel_dim)

    return normalize_image(
        (derived > threshold).to(torch.float32) if threshold else derived
    )


def image_blurr(image, ksize=3, ksigma=2, device="cpu"):
    match type(image):
        case np.ndarray:
            return numpy_image_blurr(image=image, ksize=ksize, ksigma=ksigma)
        case torch.Tensor:
            return tensor_image_blurr(
                image=image, ksize=ksize, ksigma=ksigma, device=device
            )
        case _:
            raise TypeError("Image type not supported!")


def numpy_image_blurr(image, ksize=3, ksigma=2):
    kernel = numpy_gaussian_kernel(ksize=ksize, ksigma=ksigma)

    cov = lambda img: convolve2d(img, kernel, mode="same", boundary="fill", fillvalue=0)

    match len(image.shape):
        case 2:
            return normalize_image(cov(image))
        case 3:
            channels = []
            for d in range(3):
                channels.append(image[:, :, d])
            return normalize_image(np.stack(channels, axis=-1))
        case 4:
            diff_images = np.stack(
                [
                    numpy_image_blurr(image[i], ksize=ksize, ksigma=ksigma)
                    for i in range(image.shape[0])
                ],
                axis=0,
            )
            return diff_images
        case _:
            raise ValueError("Image must be a 2D object!")


def tensor_image_blurr(image, ksize=3, ksigma=2, device="cpu"):
    image = image.to(device)
    original_shape = image.shape

    kernel = (
        tensor_gaussian_kernel(ksize=ksize, ksigma=ksigma)
        .unsqueeze(0)
        .unsqueeze(0)
        .to(device=device)
    )

    match len(image.shape):
        case 2:
            image = image.unsqueeze(0).unsqueeze(0)
        case 3:
            height, width = image.shape[1:]
            image = image.reshape(-1, 1, height, width)
        case 4:
            height, width = image.shape[2:]
            image = image.reshape(-1, 1, height, width)
        case _:
            raise ValueError("Image must be a 2D object!")

    return normalize_image(conv2d(image, kernel, padding="same").view(original_shape))


def image_sharpen(image, alpha=1, ksize=3, ksigma=2, device="cpu"):
    match type(image):
        case np.ndarray:
            return numpy_image_sharpen(image, alpha=alpha, ksize=ksize, ksigma=ksigma)
        case torch.Tensor:
            return tensor_image_sharpen(
                image, alpha=alpha, ksize=ksize, ksigma=ksigma, device=device
            )
        case _:
            raise TypeError("Image type not supported!")


def numpy_image_sharpen(image, alpha=1, ksize=3, ksigma=2):
    gaussian_kernel = numpy_gaussian_kernel(ksize=ksize, ksigma=ksigma)
    center_y, center_x = gaussian_kernel.shape[0] // 2, gaussian_kernel.shape[1] // 2
    unit_impulse = np.zeros_like(gaussian_kernel)

    unit_impulse[center_y, center_x] = 1
    sharpen_kernel = (1 + alpha) * unit_impulse - alpha * gaussian_kernel

    cov = lambda img: convolve2d(
        img, sharpen_kernel, mode="same", boundary="fill", fillvalue=0
    )

    match len(image.shape):
        case 2:
            return cov(image)
        case 3:
            return np.stack(
                [cov(image[:, :, d]) for d in range(image.shape[2])], axis=-1
            )
        case 4:
            return np.stack(
                [
                    [cov(image[i, :, :, d]) for d in range(image.shape[3])]
                    for i in range(image.shape[0])
                ],
                axis=0,
            )
        case _:
            raise ValueError("Image must be a 2D object!")


def tensor_image_sharpen(image, alpha=1, ksize=3, ksigma=2, device="cpu"):
    image = image.to(device)
    original_shape = image.shape

    gaussian_kernel = tensor_gaussian_kernel(ksize=ksize, ksigma=ksigma)
    center_y, center_x = gaussian_kernel.shape[0] // 2, gaussian_kernel.shape[1] // 2
    unit_impulse = torch.zeros_like(gaussian_kernel)

    unit_impulse[center_y, center_x] = 1
    sharpen_kernel = ((1 + alpha) * unit_impulse - alpha * gaussian_kernel).to(device)
    sharpen_kernel = sharpen_kernel.unsqueeze(0).unsqueeze(0).to(device=device)

    match len(image.shape):
        case 2:
            image = image.unsqueeze(0).unsqueeze(0)
        case 3:
            height, width = image.shape[1:]
            image = image.reshape(-1, 1, height, width)
        case 4:
            height, width = image.shape[2:]
            image = image.reshape(-1, 1, height, width)
        case _:
            raise ValueError("Image must be a 2D object!")

    return conv2d(image, sharpen_kernel, padding="same").view(original_shape)
