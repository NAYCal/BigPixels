import torch
import numpy as np

from scipy.signal import convolve2d
from torch.nn.functional import conv2d
from image_io import normalize_image
from image_math import numpy_gaussian_kernel, tensor_gaussian_kernel

def numpy_finite_difference(
    image, reduce_noise=True, out_grayscale=True, threshold=None
):
    cov = lambda img, kernel: convolve2d(
        img, kernel, mode="same", boundary="fill", fillvalue=0
    )
    
    dx = np.array([[1, 0, -1]])
    dy = dx.T
    if reduce_noise:
        gaussian_kernel = numpy_gaussian_kernel()
        dx = cov(dx, gaussian_kernel)
        dy = cov(dy, gaussian_kernel)

    match len(image.shape):
        case 2:
            image_dx = cov(image, dx)
            image_dy = cov(image, dy)
            derived = np.sqrt((image_dx**2) + (image_dy**2))
            return normalize_image((derived > threshold).astype(np.float32) if threshold else derived)
        case 3:
            channels = []
            for d in range(3):
                channel_dx = cov(image[:, :, d], dx)
                channel_dy = cov(image[:, :, d], dy)
                channels.append(np.sqrt((channel_dx**2) + (channel_dy**2)))

            derived = np.stack(channels, axis=-1)
            derived = np.mean(derived, axis=2) if out_grayscale else derived
            return normalize_image((derived > threshold).astype(np.float32) if threshold else derived)
        case 4:
            diff_images = [
                numpy_finite_difference(img, reduce_noise, out_grayscale, threshold)
                for img in image
            ]
            return diff_images
        case _:
            raise ValueError("Image must be a 2D object!")

def tensor_finite_difference(
    image, reduce_noise=True, out_grayscale=True, threshold=None, device='cpu'
):
    original_shape = image.shape
    image = image.to(device)
    dx = torch.tensor([[[[-1, 0, 1]]]], dtype=torch.float, device=device)
    dy = torch.tensor([[[[-1], [0], [1]]]], dtype=torch.float, device=device)
    if reduce_noise:
        gaussian_kernel = tensor_gaussian_kernel().unsqueeze(0).unsqueeze(0).to(device=device)
        dx = conv2d(dx, gaussian_kernel, padding='same')
        dy = conv2d(dy, gaussian_kernel, padding='same')
    
    channel_dim = None
    match len(image.shape):
        case 2:
            height, width = image.shape
            image = image.unsqueeze(0).unsqueeze(0)
        case 3:
            channel_dim = 0
            height, width = image.shape[1:]
            image = image.reshape(-1, 1, height, width)
        case 4:
            channel_dim = 1
            height, width = image.shape[2:]
        case _:
            raise ValueError("Image must be a 2D object!")
    
    image_dx = conv2d(image, dx, padding='same')
    image_dy = conv2d(image, dy, padding='same')
    derived = torch.sqrt(image_dx**2 + image_dy**2)
    derived = derived.view(original_shape)
    
    if channel_dim is not None and out_grayscale:
        derived = torch.mean(derived, dim=channel_dim)
    
    return normalize_image((derived > threshold).to(torch.float32) if threshold else derived)
