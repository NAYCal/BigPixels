import torch
import numpy as np

from tqdm import tqdm
from scipy.signal import convolve2d
from torch.nn.functional import conv2d
from image_io import normalize_image
from image_math import numpy_gaussian_kernel, tensor_gaussian_kernel

import matplotlib.pyplot as plt

DEFAULT_KERNEL_SIZE = 10
DEFAULT_KERNEL_SIGMA = 10

SUPPORTED_IMAGE_TYPE_MSG = "Only tensor/numpy ndarry objects are supported!"
NUMPY_IMAGE_SHAPE_ERROR_MSG = "Invalid image shape! Images must be in the format of either (N, H, W, C) or (H, W, C)."
TENSOR_IMAGE_SHAPE_ERROR_MSG = "Invalid image shape! Images must be in the format of either (N, C, H, W) or (C, H, W)"


def highlight_edges(
    image,
    reduce_noise=True,
    out_grayscale=True,
    binary_threshold=None,
    ksize=DEFAULT_KERNEL_SIZE,
    ksigma=DEFAULT_KERNEL_SIGMA,
    device="cpu",
):
    assert isinstance(image, np.ndarray) or isinstance(
        image, torch.Tensor
    ), SUPPORTED_IMAGE_TYPE_MSG
    
    ixx, iyy = finite_difference(
        image=image,
        reduce_noise=reduce_noise,
        out_grayscale=out_grayscale,
        binary_threshold=binary_threshold,
        ksize=ksize,
        ksigma=ksigma,
        device=device,
    )
    
    if isinstance(image, np.ndarray):
        return normalize_image(np.sqrt(ixx ** 2 + iyy ** 2))
    return normalize_image(torch.sqrt(ixx ** 2 + iyy ** 2))
            

def finite_difference(
    image,
    reduce_noise=True,
    out_grayscale=True,
    binary_threshold=None,
    ksize=DEFAULT_KERNEL_SIZE,
    ksigma=DEFAULT_KERNEL_SIGMA,
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
                binary_threshold=binary_threshold,
            )
        case torch.Tensor:
            return tensor_finite_difference(
                image,
                reduce_noise=reduce_noise,
                out_grayscale=out_grayscale,
                ksize=ksize,
                ksigma=ksigma,
                binary_threshold=binary_threshold,
                device=device,
            )
        case _:
            raise TypeError(SUPPORTED_IMAGE_TYPE_MSG)


def numpy_finite_difference(
    image,
    reduce_noise=True,
    out_grayscale=True,
    ksize=DEFAULT_KERNEL_SIZE,
    ksigma=DEFAULT_KERNEL_SIGMA,
    binary_threshold=None,
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
        case 3:
            channels_dx = []
            channels_dy = []
            for d in range(image.shape[2]):
                channels_dx.append(cov(image[:, :, d], dx))
                channels_dy.append(cov(image[:, :, d], dy))
                
            ixx = np.stack(channels_dx, axis=-1)
            iyy = np.stack(channels_dy, axis=-1)
            
            ixx = np.mean(ixx, axis=-1) if out_grayscale else ixx
            iyy = np.mean(iyy, axis=-1) if out_grayscale else iyy
            
            ixx = (ixx > binary_threshold).astype(np.float32) if binary_threshold else ixx
            iyy = (iyy > binary_threshold).astype(np.float32) if binary_threshold else iyy
            return ixx, iyy
        case 4:
            ixx_list = []
            iyy_list = []
            for img in image:
                ixx, iyy = numpy_finite_difference(
                    image=img,
                    reduce_noise=reduce_noise,
                    out_grayscale=out_grayscale,
                    binary_threshold=binary_threshold
                )
                ixx_list.append(ixx)
                iyy_list.append(iyy)
            
            return np.stack(ixx_list, axis=0), np.stack(iyy_list, axis=0)
        case _:
            raise ValueError(NUMPY_IMAGE_SHAPE_ERROR_MSG)


def tensor_finite_difference(
    image,
    reduce_noise=True,
    out_grayscale=True,
    ksize=DEFAULT_KERNEL_SIZE,
    ksigma=DEFAULT_KERNEL_SIGMA,
    binary_threshold=None,
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
        case 3:
            channel_dim = 0
            height, width = image.shape[1:]
            image = image.reshape(-1, 1, height, width)
        case 4:
            channel_dim = 1
            height, width = image.shape[2:]
            image = image.reshape(-1, 1, height, width)
        case _:
            raise ValueError(TENSOR_IMAGE_SHAPE_ERROR_MSG)
        
    ixx = conv2d(image, dx, padding="same").view(original_shape)
    iyy = conv2d(image, dy, padding="same").view(original_shape)

    if channel_dim is not None and out_grayscale:
        ixx = torch.mean(ixx, dim=channel_dim)
        iyy = torch.mean(iyy, dim=channel_dim)
        
    if binary_threshold:
        ixx = (ixx > binary_threshold).to(torch.float32)
        iyy = (iyy > binary_threshold).to(torch.float32)

    return ixx, iyy


def image_blurr(
    image, ksize=DEFAULT_KERNEL_SIZE, ksigma=DEFAULT_KERNEL_SIGMA, device="cpu"
):
    match type(image):
        case np.ndarray:
            return numpy_image_blurr(image=image, ksize=ksize, ksigma=ksigma)
        case torch.Tensor:
            return tensor_image_blurr(
                image=image, ksize=ksize, ksigma=ksigma, device=device
            )
        case _:
            raise TypeError(SUPPORTED_IMAGE_TYPE_MSG)


def numpy_image_blurr(image, ksize=DEFAULT_KERNEL_SIZE, ksigma=DEFAULT_KERNEL_SIGMA):
    kernel = numpy_gaussian_kernel(ksize=ksize, ksigma=ksigma)
    cov = lambda img: convolve2d(img, kernel, mode="same", boundary="fill", fillvalue=0)

    match len(image.shape):
        case 3:
            channels = []
            for d in range(3):
                channels.append(cov(image[:, :, d]))
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
            raise ValueError(NUMPY_IMAGE_SHAPE_ERROR_MSG)


def tensor_image_blurr(
    image, ksize=DEFAULT_KERNEL_SIZE, ksigma=DEFAULT_KERNEL_SIGMA, device="cpu"
):
    image = image.to(device)
    original_shape = image.shape

    kernel = (
        tensor_gaussian_kernel(ksize=ksize, ksigma=ksigma)
        .unsqueeze(0)
        .unsqueeze(0)
        .to(device=device)
    )

    match len(image.shape):
        case 3:
            height, width = image.shape[1:]
            image = image.reshape(-1, 1, height, width)
        case 4:
            height, width = image.shape[2:]
            image = image.reshape(-1, 1, height, width)
        case _:
            raise ValueError(TENSOR_IMAGE_SHAPE_ERROR_MSG)

    return normalize_image(conv2d(image, kernel, padding="same").view(original_shape))


def image_sharpen(
    image, alpha=1, ksize=DEFAULT_KERNEL_SIZE, ksigma=DEFAULT_KERNEL_SIGMA, device="cpu"
):
    match type(image):
        case np.ndarray:
            return numpy_image_sharpen(image, alpha=alpha, ksize=ksize, ksigma=ksigma)
        case torch.Tensor:
            return tensor_image_sharpen(
                image, alpha=alpha, ksize=ksize, ksigma=ksigma, device=device
            )
        case _:
            raise TypeError(SUPPORTED_IMAGE_TYPE_MSG)


def numpy_image_sharpen(
    image, alpha=1, ksize=DEFAULT_KERNEL_SIZE, ksigma=DEFAULT_KERNEL_SIGMA
):
    gaussian_kernel = numpy_gaussian_kernel(ksize=ksize, ksigma=ksigma)
    center_y, center_x = gaussian_kernel.shape[0] // 2, gaussian_kernel.shape[1] // 2
    unit_impulse = np.zeros_like(gaussian_kernel)

    unit_impulse[center_y, center_x] = 1
    sharpen_kernel = (1 + alpha) * unit_impulse - alpha * gaussian_kernel

    cov = lambda img: convolve2d(
        img, sharpen_kernel, mode="same", boundary="fill", fillvalue=0
    )

    match len(image.shape):
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
            raise ValueError(NUMPY_IMAGE_SHAPE_ERROR_MSG)


def tensor_image_sharpen(
    image, alpha=1, ksize=DEFAULT_KERNEL_SIZE, ksigma=DEFAULT_KERNEL_SIGMA, device="cpu"
):
    image = image.to(device)
    original_shape = image.shape

    gaussian_kernel = tensor_gaussian_kernel(ksize=ksize, ksigma=ksigma)
    center_y, center_x = gaussian_kernel.shape[0] // 2, gaussian_kernel.shape[1] // 2
    unit_impulse = torch.zeros_like(gaussian_kernel)

    unit_impulse[center_y, center_x] = 1
    sharpen_kernel = ((1 + alpha) * unit_impulse - alpha * gaussian_kernel).to(device)
    sharpen_kernel = sharpen_kernel.unsqueeze(0).unsqueeze(0).to(device=device)

    match len(image.shape):
        case 3:
            height, width = image.shape[1:]
            image = image.reshape(-1, 1, height, width)
        case 4:
            height, width = image.shape[2:]
            image = image.reshape(-1, 1, height, width)
        case _:
            raise ValueError(TENSOR_IMAGE_SHAPE_ERROR_MSG)

    return conv2d(image, sharpen_kernel, padding="same").view(original_shape)


def hybrid_images(
    image_to_highpass,
    image_to_lowpass,
    ksize=DEFAULT_KERNEL_SIZE,
    ksigma=DEFAULT_KERNEL_SIGMA,
    device="cpu",
):
    assert type(image_to_highpass) == type(image_to_lowpass), "Image types must match!"
    assert image_to_highpass.shape == image_to_lowpass.shape, "Image sizes must match!"
    highpass_image = image_to_highpass - image_blurr(
        image_to_highpass, ksize=ksize, ksigma=ksigma, device=device
    )
    lowpass_image = image_blurr(image_to_lowpass)

    if isinstance(highpass_image, torch.Tensor):
        highpass_image = highpass_image.to("cpu")
        lowpass_image = lowpass_image.to("cpu")

    return highpass_image + lowpass_image


def gaussian_stack(
    image,
    depth=5,
    ksize=DEFAULT_KERNEL_SIZE,
    ksigma=DEFAULT_KERNEL_SIGMA,
    device="cpu",
    is_batch_memory=False,
):
    assert isinstance(image, np.ndarray) or isinstance(
        image, torch.Tensor
    ), SUPPORTED_IMAGE_TYPE_MSG
    image = (
        image.to(device)
        if isinstance(image, torch.Tensor) and is_batch_memory
        else image
    )
    g_stack = [image]
    curr = image
    for _ in tqdm(range(depth - 1)):
        curr = image_blurr(curr, ksize=ksize, ksigma=ksigma, device=device)
        curr = (
            curr.to("cpu")
            if not is_batch_memory and isinstance(image, torch.Tensor)
            else curr
        )
        g_stack.append(curr)

    match (len(image.shape)):
        case 3:
            return (
                torch.stack(g_stack, dim=0).to(device)
                if isinstance(image, torch.Tensor)
                else np.stack(g_stack, axis=0)
            )
        case 4:
            return (
                torch.stack(g_stack, dim=1).to(device)
                if isinstance(image, torch.Tensor)
                else np.stack(g_stack, axis=1)
            )
        case _:
            raise ValueError("Invalid image!")


def laplacian_stack(
    image,
    depth=5,
    ksize=DEFAULT_KERNEL_SIZE,
    ksigma=DEFAULT_KERNEL_SIGMA,
    device="cpu",
    is_batch_memory=False,
):
    assert isinstance(image, np.ndarray) or isinstance(
        image, torch.Tensor
    ), SUPPORTED_IMAGE_TYPE_MSG
    g_stack = gaussian_stack(
        image,
        depth=depth,
        ksize=ksize,
        ksigma=ksigma,
        device=device,
        is_batch_memory=is_batch_memory,
    )

    if is_batch_memory and isinstance(g_stack, torch.Tensor):
        g_stack = g_stack.to("cpu")

    match (len(g_stack.shape)):
        case 4:
            l_stack = g_stack[:-1] - g_stack[1:]
            if isinstance(l_stack, np.ndarray):
                l_stack = np.concatenate([l_stack, g_stack[-1][np.newaxis]], axis=0)
            else:
                l_stack = torch.concat([l_stack, g_stack[-1].unsqueeze(0)], dim=0).to(
                    device
                )
            return l_stack
        case 5:
            l_stack = g_stack[:, :-1] - g_stack[:, 1:]
            if isinstance(l_stack, np.ndarray):
                l_stack = np.concatenate(
                    [l_stack, g_stack[:, -1][:, np.newaxis]], axis=1
                )
            else:
                l_stack = torch.concat(
                    [l_stack, g_stack[:, -1].unsqueeze(1)], dim=1
                ).to(device)
            return l_stack
        case _:
            raise ValueError("Invalid image!")


def collapse_stack(stack):
    assert isinstance(stack, np.ndarray) or isinstance(
        stack, torch.Tensor
    ), SUPPORTED_IMAGE_TYPE_MSG
    match (len(stack.shape)):
        case 4:
            if isinstance(stack, np.ndarray):
                return stack.sum(axis=0)
            else:
                return stack.sum(dim=0)
        case 5:
            if isinstance(stack, np.ndarray):
                return stack.sum(axis=1)
            else:
                return stack.sum(dim=1)
        case _:
            raise ValueError(
                "Invalid stack! Stack must be have the same format as (N, D, Image.shape)!"
            )


def image_blend(
    images,
    masks,
    depth=5,
    ksize=DEFAULT_KERNEL_SIZE,
    ksigma=DEFAULT_KERNEL_SIGMA,
    device="cpu",
    is_batch_memory=False,
):
    if not (
        isinstance(images, (np.ndarray, torch.Tensor))
        and isinstance(masks, (np.ndarray, torch.Tensor))
    ):
        raise TypeError(SUPPORTED_IMAGE_TYPE_MSG)

    if type(images) != type(masks):
        raise TypeError(
            f"Both images and masks must have the same type! Images: {type(images)}, Masks: {type(masks)}"
        )

    if len(images.shape) != 4 or len(masks.shape) != 4:
        raise ValueError(f"Both images and masks must have (N, Image.shape) shapes!")

    l_stack = laplacian_stack(
        image=images,
        depth=depth,
        ksize=ksize,
        ksigma=ksigma,
        device=device,
        is_batch_memory=is_batch_memory,
    )

    mask_stack = gaussian_stack(
        image=masks,
        depth=depth,
        ksize=ksize,
        ksigma=ksigma,
        device=device,
        is_batch_memory=is_batch_memory,
    )

    result_stack = l_stack * mask_stack
    if isinstance(result_stack, torch.Tensor):
        return collapse_stack(torch.sum(result_stack, dim=0))
    return collapse_stack(np.sum(result_stack, axis=0))
