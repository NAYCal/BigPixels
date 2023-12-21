""" Inspired by CS 180/280 at UC Berkeley.
"""
import tqdm
import numpy as np


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
        self.red_channel = gp_image[channel_height : channel_height * 2]
        self.green_channel = gp_image[:-channel_height]

        self.auto_crop()
        
        match base_channel:
            case "red":
                self.base_channel = self.red_channel
                self.target_channels = [self.green_channel, self.blue_channel]
            case "blue":
                self.base_channel = self.blue_channel
                self.target_channels = [self.red_channel, self.green_channel]
            case _:
                self.base_channel = self.green_channel
                self.target_channels = [self.red_channel, self.blue_channel]

    def auto_crop(self):
        self.blue_channel = self.blue_channel[20:-20, 20:-20]
        self.red_channel = self.red_channel[20:-20, 20:-20]
        self.green_channel = self.green_channel[20:-20, 20:-20]


def exhausive_alignment(
    base, target, center_x=None, center_y=None, x_range=15, y_range=15
):
    pass


def pyramid_alignment(
    base, target, center_x=None, center_y=None, x_range=15, y_range=15
):
    pass
