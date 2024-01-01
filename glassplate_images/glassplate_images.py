""" Inspired by CS 180/280 at UC Berkeley.
"""
import numpy as np
from scipy import signal

from tqdm import tqdm
from functional.image_math import ssd, numpy_gaussian_kernel


class GlassPlateImages:
    CHANNELS = {"blue", "green", "red"}

    def __init__(self, gp_image, base_channel="red") -> None:
        assert isinstance(gp_image, np.ndarray), "Input image must be in numpy format!"
        assert (
            base_channel.lower() in self.CHANNELS
        ), f"Base channel must be in {self.CHANNELS}"
        height = gp_image.shape[0]
        channel_height = height // 3

        self.gp_image = gp_image
        self.blue_channel = gp_image[:channel_height]
        self.green_channel = gp_image[channel_height : channel_height * 2]
        self.red_channel = (
            gp_image[channel_height * 2 :]
            if height % 3 == 0
            else gp_image[channel_height * 2 : -(height % 3)]
        )

        self.base_channel = base_channel
        self.auto_crop()

    def auto_crop(self):
        self.blue_channel = self.blue_channel[20:-20, 20:-20]
        self.red_channel = self.red_channel[20:-20, 20:-20]
        self.green_channel = self.green_channel[20:-20, 20:-20]

    def align(self, x_range=30, y_range=30):
        match self.base_channel:
            case "red":
                red_channel = self.red_channel
                green_channel, _ = pyramid_alignment(
                    self.red_channel, self.green_channel, x_range, y_range
                )
                blue_channel, _ = pyramid_alignment(
                    self.red_channel, self.blue_channel, x_range, y_range
                )
            case "blue":
                red_channel, _ = pyramid_alignment(
                    self.blue_channel, self.red_channel, x_range, y_range
                )
                green_channel, _ = pyramid_alignment(
                    self.blue_channel, self.green_channel, x_range, y_range
                )
                blue_channel = self.blue_channel
            case _:
                red_channel, _ = pyramid_alignment(
                    self.green_channel, self.red_channel, x_range, y_range
                )
                blue_channel, _ = pyramid_alignment(
                    self.green_channel, self.blue_channel, x_range, y_range
                )
                green_channel = self.green_channel

        return np.dstack([red_channel, green_channel, blue_channel])


def exhausive_alignment(base, target, x_range=30, y_range=30):
    best_target, best_loss = target, float("inf")
    best_offset = (0, 0)
    for offset_y in tqdm(range(-y_range, y_range)):
        for offset_x in range(-x_range, x_range):
            shifted_target = np.roll(target, (offset_x, offset_y), (1, 0))
            loss = ssd(shifted_target, base)

            if best_loss > loss:
                best_loss = loss
                best_target = shifted_target
                best_offset = (offset_x, offset_y)

    return best_target, best_offset


def pyramid_alignment(
    base,
    target,
    x_range=30,
    y_range=30,
    min_size=100,
    gaussian_kernel=numpy_gaussian_kernel(),
):
    assert min_size >= 0, "Min size must be a positive number!"
    if base.shape[0] <= min_size or base.shape[1] <= min_size:
        return exhausive_alignment(base, target, x_range, y_range)

    resized_base = signal.convolve2d(base, gaussian_kernel, mode="same")[::2, ::2]
    resized_target = signal.convolve2d(target, gaussian_kernel, mode="same")[::2, ::2]

    _, best_offset = pyramid_alignment(
        resized_base, resized_target, x_range, y_range, min_size, gaussian_kernel
    )

    best_offset_x, best_offset_y = best_offset[0] * 2, best_offset[1] * 2
    best_target = np.roll(target, (best_offset_x, best_offset_y), (1, 0))
    return exhausive_alignment(base, best_target, x_range, y_range)
