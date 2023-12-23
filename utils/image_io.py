import os
import torch
import warnings
import numpy as np
from PIL import Image
from torchvision import transforms

IMAGE_FILE_TYPES = (".jpg", ".jpeg", ".png", ".bmp", ".tif")

def read_image(file_name, directory_path="data", data_format="torch", out_size=None):
    assert file_name.lower().endswith(
        IMAGE_FILE_TYPES
    ), f"File name must end with one of the following: {IMAGE_FILE_TYPES}"
    assert data_format.lower() in {"torch", "numpy", "pil"}

    image_path = os.path.join(directory_path, file_name)
    image = Image.open(image_path)

    if out_size is not None:
        image = image.resize(out_size)

    match data_format.lower():
        case "torch":
            converted_image = transforms.ToTensor()(image)
        case "numpy":
            converted_image = normalize_image(np.array(image).astype(np.float32))
        case "pil":
            converted_image = image

    return converted_image


def read_all_images(directory_path="data", data_format="torch", out_size=None):
    image_paths = []
    file_names = []
    for root, _, files in os.walk(directory_path):
        for file in files:
            if file.lower().endswith(IMAGE_FILE_TYPES):
                image_path = os.path.join(root, file)
                image_paths.append(image_path)
                file_names.append(file)

    images = []
    for image_path in image_paths:
        images.append(read_image(image_path, "", data_format=data_format, out_size=out_size))

    return images, file_names


def save_image(img_rep, file_name, directory_path="out", out_size=None):
    assert file_name.lower().endswith(
        IMAGE_FILE_TYPES
    ), f"File name must end with one of the following: {IMAGE_FILE_TYPES}"
    os.makedirs(directory_path, exist_ok=True)
    save_path = os.path.join(directory_path, file_name)

    match type(img_rep):
        case np.ndarray:
            uarray = img_rep * 255 if img_rep.max() <= 1.0 else img_rep
            uarray = uarray.astype(np.uint8)
            image = Image.fromarray(uarray)
        case torch.Tensor:
            uarray = img_rep * 255 if img_rep.max() <= 1.0 else img_rep
            uarray = uarray.numpy().transpose(1, 2, 0).astype(np.uint8)
            image = Image.fromarray(uarray)
        case _:
            image = img_rep

    if out_size is not None:
        image = image.resize(out_size)
    image.save(save_path)


def normalize_image(image):
    if not isinstance(image, np.ndarray) and not isinstance(image, torch.Tensor):
        warnings.warn("WARNING: Image is not a ndarray or tensor")
        return image

    min_val = image.min()
    max_val = image.max()

    return (image - min_val) / (max_val - min_val)
