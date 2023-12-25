import torch
import numpy as np

from image_math import numpy_gaussian_kernel, tensor_gaussian_kernel
from scipy.signal import convolve2d
from torch.nn.functional import conv2d


# TODO: Add thresholding
def numpy_finite_difference(
    image, reduce_noise=True, out_grayscale=True, threshold=None
):
    dx = np.array([[1, 0, -1]])
    dy = dx.T

    if reduce_noise:
        gaussian_kernel = numpy_gaussian_kernel()
        dx = convolve2d(dx, gaussian_kernel)
        dy = convolve2d(dy, gaussian_kernel)

    cov = lambda img, kernel: convolve2d(
        img, kernel, mode="same", boundary="fill", fillvalue=0
    )

    match len(image.shape):
        case 2:
            image_dx = cov(image, dx)
            image_dy = cov(image, dy)
            derived = np.sqrt((image_dx**2) + (image_dy**2))
            return (derived > threshold).astype(np.float32) if threshold else derived
        case 3:
            channels = []
            for d in range(3):
                channel_dx = cov(image[:, :, d], dx)
                channel_dy = cov(image[:, :, d], dy)
                channels.append(np.sqrt((channel_dx**2) + (channel_dy**2)))

            derived = np.stack(channels, axis=-1)
            derived = np.mean(derived, axis=2) if out_grayscale else derived
            return (derived > threshold).astype(np.float32) if threshold else derived
        case 4:
            diff_images = [
                numpy_finite_difference(img, reduce_noise, out_grayscale, threshold)
                for img in image
            ]
            return diff_images
        case _:
            raise ValueError("Image must be a 2D object!")
