import matplotlib.pyplot as plt
from glassplate_images import GlassPlateImages

from utils import image_io

if __name__ == "__main__":
    sample_image = image_io.read_image("cathedral.jpg", "samples/glassplate_images", "numpy")
    sample_channel = GlassPlateImages(sample_image).red_channel
    plt.imshow(sample_channel, cmap='gray')
    plt.show()