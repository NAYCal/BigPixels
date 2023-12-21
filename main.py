import matplotlib.pyplot as plt
from glassplate_images import GlassPlateImages

from utils import image_io

if __name__ == "__main__":
    sample_image = image_io.read_image("cathedral.jpg", "samples/glassplate_images", "numpy")
    sample_gpi = GlassPlateImages(sample_image)
    plt.imshow(sample_gpi.align(), cmap='gray')
    plt.show()