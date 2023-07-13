import torch
import numpy as np

import numpy as np
import os
from matplotlib import pyplot as plt
from torchvision.transforms import transforms


def plot(inputs, targets, predictions, path, update):
    """Plotting the inputs, targets and predictions to file ``path``."""
    os.makedirs(path, exist_ok=True)
    fig, axes = plt.subplots(ncols=3, figsize=(15, 5))

    for i in range(len(inputs)):
        for ax, data, title in zip(axes, [inputs, targets, predictions], ["Input", "Target", "Prediction"]):
            ax.clear()
            ax.set_title(title)
            ax.imshow(data[i, 0], cmap="gray", interpolation="none")
            ax.set_axis_off()
        fig.savefig(os.path.join(path, f"{update:07d}_{i:02d}.png"), dpi=100)

    plt.close(fig)


def linearize_color_channel(c: float) -> float:
    if c <= 0.04045:
        return c / 12.92
    else:
        return ((c + 0.055) / 1.055) ** 2.4


def linearize_gray_channel(y: float) -> float:
    if y <= 0.0031308:
        return y * 12.92
    else:
        return 1.055 * y ** (1 / 2.4) - 0.055


linearize_color_channel = np.vectorize(linearize_color_channel)
linearize_gray_channel = np.vectorize(linearize_gray_channel)


def to_grayscale(pil_image: np.ndarray) -> np.ndarray:
    if len(pil_image.shape) == 2:
        return pil_image.copy().reshape((1, pil_image.shape[0], pil_image.shape[1]))

    if len(pil_image.shape) != 3:
        raise ValueError(f"The image inserted, has {len(pil_image.shape)} dimensions")

    if pil_image.shape[2] != 3:
        raise ValueError(f"The third dimension does not have a size of exactly three, but {pil_image.shape[2]}")

    coefficients = [0.2126, 0.7152, 0.0722]

    image_copy = pil_image.copy().astype(np.float32)
    image_copy /= 255

    for i in range(3):
        image_copy[:, :, i] = linearize_color_channel(image_copy[:, :, i])
        image_copy[:, :, i] *= coefficients[i]

    image_copy = np.sum(image_copy, axis=2).reshape((1, pil_image.shape[0], pil_image.shape[1]))
    image_copy = linearize_gray_channel(image_copy)
    image_copy *= 255

    return np.round(image_copy).astype(pil_image.dtype)


def prepare_image(
        image: np.ndarray,
        x: int,
        y: int,
        width: int,
        height: int,
        size: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(image.shape) != 3 or image.shape[0] != 1:
        raise ValueError("Image shape must be 3D with exactly 1 channel (shape: (1, H, W)).")

    if width < 2 or height < 2 or size < 2:
        raise ValueError("Width, height, and size must be at least 2.")

    if x < 0 or x + width > image.shape[2]:
        raise ValueError("Pixelated area would exceed input image width.")

    if y < 0 or y + height > image.shape[1]:
        raise ValueError("Pixelated area would exceed input image height.")

    pixelated_image = np.copy(image)
    known_array = np.ones_like(image, dtype=bool)
    target_array = np.copy(image[:, y:y + height, x:x + width])

    for row in range(y, y + height, size):
        for col in range(x, x + width, size):
            block_width = min(size, x + width - col)
            block_height = min(size, y + height - row)

            block = image[:, row:row + block_height, col:col + block_width]
            block_mean = np.mean(block, axis=(1, 2), keepdims=True)

            pixelated_image[:, row:row + block_height, col:col + block_width] = block_mean
            known_array[:, row:row + block_height, col:col + block_width] = False

    return pixelated_image, known_array, target_array


def stack_with_padding(batch_as_list: list):
    max_values = []
    for values in zip(*[image[0].shape for image in batch_as_list]):
        max_values.append(max(values))
    shape = (len(batch_as_list),) + tuple(max_values)

    stacked_full_images = []
    stacked_known_arrays = []
    target_arrays = []
    image_files = []

    for item in batch_as_list:
        full_image, known_array, target_array, image_file = item

        padded_full_image = np.zeros(shape[1:], dtype=full_image.dtype)
        padded_full_image[:, :full_image.shape[1], :full_image.shape[2]] = full_image
        stacked_full_images.append(padded_full_image)

        # takes shape of full image instead of known array
        # padded_known_array = np.zeros(shape[1:], dtype=known_array.dtype)
        # padded_known_array[:, :known_array.shape[1], :known_array.shape[2]] = known_array

        stacked_known_arrays.append(known_array)

        target_arrays.append(torch.tensor(target_array))
        image_files.append(image_file)

    stacked_full_images = torch.tensor(np.array(stacked_full_images))
    stacked_known_arrays = torch.tensor(np.array(stacked_known_arrays))

    return stacked_full_images, stacked_known_arrays, target_arrays, image_files
