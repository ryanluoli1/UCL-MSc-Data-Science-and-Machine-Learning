import os
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageOps
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def pad_and_resize_image(image, target_size=256, type='image'):
    width, height = image.size

    # Calculate the new dimensions while preserving the aspect ratio
    if width > height:
        new_width = target_size
        new_height = int(height * (target_size / width))
    else:
        new_height = target_size
        new_width = int(width * (target_size / height))

    # Resize the image
    resized_image = image.resize((new_width, new_height), Image.ANTIALIAS)

    # Calculate the padding
    left_padding = 0
    top_padding = 0
    right_padding = target_size - new_width
    bottom_padding = target_size - new_height

    padding = (left_padding, top_padding, right_padding, bottom_padding)

    if type == 'image':
        # Pad the image using the edge pixel value
        padded_image = ImageOps.expand(resized_image, padding,
                                       fill=resized_image.getpixel((new_width - 6, new_height - 6)))
    elif type == 'label':
        # Pad the label using background pixel value (2)
        padded_image = ImageOps.expand(resized_image, padding, fill=2)

    padded_array = np.array(padded_image)

    return padded_array, new_height, new_width


def process_images(input_folder, label_folder, valid_extensions=("jpg", "jpeg", "png", "bmp", "tiff", "gif")):
    results = []
    image_array = []
    label_array = []
    size = []

    for file in os.listdir(input_folder):
        if os.path.isfile(os.path.join(input_folder, file)) and file.lower().endswith(valid_extensions):
            # Get the input image file path and name without extension
            input_file_path = os.path.join(input_folder, file)
            input_file_name = os.path.splitext(file)[0]

            # Find the corresponding label image file path and name
            label_file_name = input_file_name + ".png"
            label_file_path = os.path.join(label_folder, label_file_name)

            # Make sure the corresponding label image file exists
            if not os.path.isfile(label_file_path):
                print(f"Label image file {label_file_name} not found for input image file {input_file_name}")
                continue

            # Load the input and label images
            with Image.open(input_file_path) as input_img, Image.open(label_file_path) as label_img:
                # Convert the input image to RGB mode
                input_img_rgb = input_img.convert("RGB")
                input_padded_array, width, height = pad_and_resize_image(input_img_rgb)

                # Resize and pad the label image with background pixel value (2)
                label_padded_array, _, _ = pad_and_resize_image(label_img, type='label')

                # Add the padded input and label arrays to their respective lists
                image_array.append(input_padded_array)
                label_array.append(label_padded_array)
                size.append([width, height])

    return image_array, label_array, size


if __name__ == '__main__':
    input_folder = "images"
    label_folder = "trimaps"

    img, label, size = process_images(input_folder, label_folder)

    img = np.array(img)
    label = np.array(label)
    size = np.array(size)

    np.save("./img.npy", img)
    np.save("./label.npy", label)
    np.save("./size.npy", size)
