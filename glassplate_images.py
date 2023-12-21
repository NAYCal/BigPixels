""" Inspired by CS 180/280 at UC Berkeley.
"""
import tqdm
import numpy as np
from utils.image_math import ssd


class GlassPlateImages:
    CHANNELS = {"blue", "green", "red"}

    def __init__(self, gp_image, base_channel="green") -> None:
        assert isinstance(gp_image, np.ndarray), "Input image must be in numpy format!"
        assert (
            base_channel.lower() in self.CHANNELS
        ), f"Base channel must be in {self.CHANNELS}"
        height = gp_image.shape[0]
        channel_height = height // 3

        self.gp_image = gp_image
        self.blue_channel = gp_image[:channel_height]
        self.green_channel = gp_image[channel_height : channel_height * 2]
        self.red_channel = gp_image[channel_height * 2:] if height % 3 == 0 else gp_image[channel_height * 2:-1]
        
        print(self.blue_channel.shape, self.red_channel.shape, self.green_channel.shape)

        self.base_channel = base_channel
        self.auto_crop()

    def auto_crop(self):
        self.blue_channel = self.blue_channel[20:-20, 20:-20]
        self.red_channel = self.red_channel[20:-20, 20:-20]
        self.green_channel = self.green_channel[20:-20, 20:-20]

    def align(self):
        match self.base_channel:
            case "red":
                red_channel = self.red_channel
                green_channel, offset = exhausive_alignment(self.red_channel, self.green_channel)
                blue_channel, offset = exhausive_alignment(self.red_channel, self.blue_channel)
            case "blue":
                red_channel, offset = exhausive_alignment(self.blue_channel, self.red_channel)
                green_channel, offset = exhausive_alignment(self.blue_channel, self.green_channel)
                blue_channel = self.blue_channel
            case _:
                red_channel, offset = exhausive_alignment(self.green_channel, self.red_channel)
                blue_channel, offset = exhausive_alignment(self.green_channel, self.blue_channel)
                green_channel = self.green_channel
                
        return np.dstack([red_channel, green_channel, blue_channel])


def exhausive_alignment(base, target, x_range=15, y_range=15):
    best_target, best_loss = target, float("inf")
    best_offset = (0, 0)
    for offset_y in range(-y_range, y_range):
        for offset_x in range(-x_range, x_range):
            shifted_target = np.roll(target, (offset_y, offset_x), (0, 1))
            loss = ssd(shifted_target, base)

            if best_loss > loss:
                best_loss = loss
                best_target = shifted_target
                best_offset = (offset_y, offset_x)

    return best_target, best_offset


def pyramid_alignment(
    base, target, center_x=None, center_y=None, x_range=15, y_range=15
):
    pass
