import os
import torch
import warnings
import numpy as np
from PIL import Image
from torchvision import transforms

SUPPORTED_IMAGE_FILE_TYPES = (".jpg", ".jpeg", ".png", ".bmp", ".tif")


def read_image(
    file_name,
    directory_path="data",
    data_format="torch",
    out_size=None,
    grayscale=False,
):
    assert file_name.lower().endswith(
        SUPPORTED_IMAGE_FILE_TYPES
    ), f"File name must end with one of the following: {SUPPORTED_IMAGE_FILE_TYPES}"
    assert data_format.lower() in {
        "tensor",
        "torch",
        "numpy",
        "np",
        "pil",
        "pillow",
    }, "Data format must be valid!"

    parent_directory = os.path.dirname(os.getcwd())
    image_path = os.path.join(parent_directory, directory_path, file_name)
    image = adjust_pillow_image(Image.open(image_path))

    if grayscale:
        image = image.convert("L")

    if out_size is not None:
        image = image.resize(out_size)

    match data_format.lower():
        case "torch" | "tensor":
            converted_image = transforms.ToTensor()(image)
        case "numpy" | "np":
            converted_image = normalize_image(np.array(image).astype(np.float32))
        case "pil" | "pillow":
            converted_image = image

    return converted_image


def read_all_images(
    directory_path="data", data_format="torch", out_size=None, grayscale=False
):
    parent_directory = os.path.dirname(os.getcwd())
    directory_path = os.path.join(parent_directory, directory_path)

    image_paths = []
    file_names = []
    for root, _, files in os.walk(directory_path):
        for file in files:
            if file.lower().endswith(SUPPORTED_IMAGE_FILE_TYPES):
                image_path = os.path.join(root, file)
                image_paths.append(image_path)
                file_names.append(file)

    images = []
    for image_path in image_paths:
        images.append(
            read_image(
                image_path,
                "",
                data_format=data_format,
                out_size=out_size,
                grayscale=grayscale,
            )
        )

    match data_format:
        case "torch" | "tensor":
            images = torch.stack(images, dim=0)
        case "numpy" | "np":
            images = np.stack(images, axis=0)

    return images, file_names


def save_image(img_rep, file_name, directory_path="out", out_size=None):
    assert file_name.lower().endswith(
        SUPPORTED_IMAGE_FILE_TYPES
    ), f"File name must end with one of the following: {SUPPORTED_IMAGE_FILE_TYPES}"
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


def adjust_pillow_image(image):
    exif = image._getexif()
    if exif:
        orientation_tag = 274
        if orientation_tag in exif:
            orientation = exif[orientation_tag]

            if orientation == 3:
                image = image.rotate(180)
            elif orientation == 6:
                image = image.rotate(-90, expand=True)
            elif orientation == 8:
                image = image.rotate(90, expand=True)

    return image
