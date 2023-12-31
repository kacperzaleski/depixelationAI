import glob
import pickle
import dill
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset
import os
from PIL import Image
from torchvision.transforms import transforms

from utils import to_grayscale
from utils import prepare_image


class RandomImagePixelationDataset(Dataset):
    def __init__(self,
                 image_dir,
                 width_range: tuple[int, int],
                 height_range: tuple[int, int],
                 size_range: tuple[int, int],
                 dtype: Optional[type] = None
                 ):
        self.width_range = width_range
        self.height_range = height_range
        self.size_range = size_range
        self.dtype = dtype
        # collect image absolute paths
        self.images = []
        self.image_dir = os.path.abspath(image_dir)
        filelist = glob.glob(os.path.join(image_dir, "**", "*"), recursive=True)
        self.images = [f for f in filelist if (f.endswith(".jpg") or f.endswith(".JPG")) and os.path.isfile(f)]
        self.images.sort()

        self.resize_transforms = transforms.Compose([
            transforms.Resize(size=64),
            transforms.CenterCrop(size=(64, 64)),
        ])

        # check ranges
        if width_range[0] < 2 or height_range[0] < 2 or size_range[0] < 2:
            raise ValueError('Minimum value is smaller than 2')
        if width_range[0] > width_range[1] or height_range[0] > height_range[1] or size_range[0] > size_range[1]:
            raise ValueError('Minimum value is greater than maximum value')

    def __getitem__(self, index):
        img = Image.open(self.images[index])
        img = self.resize_transforms(img)
        npimg = np.array(img, dtype=self.dtype)
        grayscale_img = to_grayscale(npimg)
        generator = np.random.default_rng(index)
        rnd_width = generator.integers(low=self.width_range[0], high=self.width_range[1], endpoint=True)
        rnd_height = generator.integers(low=self.height_range[0],
                                        high=self.height_range[1],
                                        endpoint=True)
        image_width, image_height = img.size
        width = min(image_width, rnd_width)
        height = min(image_height, rnd_height)

        rnd_x = generator.integers(low=0, high=npimg.shape[1] - width, endpoint=True)
        rnd_y = generator.integers(low=0, high=npimg.shape[0] - height, endpoint=True)

        rnd_size = generator.integers(low=self.size_range[0], high=self.size_range[1])

        pixelated_image, known_array, target_array = prepare_image(grayscale_img, rnd_x, rnd_y, rnd_width,
                                                                   rnd_height, rnd_size)
        pixelated_image = (pixelated_image / 255).astype(np.float32)
        target_array = (target_array / 255).astype(np.float32)
        full_image = np.concatenate((pixelated_image, known_array), axis=0)

        return full_image, known_array, target_array, self.images[index]

    def __len__(self):
        return len(self.images)


class ImageDepixelationTestDataset(Dataset):
    def __init__(self, data_path):
        with open(data_path, 'rb') as file:
            data_dict = pickle.load(file)

        self.pixelated_images = data_dict['pixelated_images']
        self.known_arrays = data_dict['known_arrays']

    def __getitem__(self, index):
        pixelated_image = self.pixelated_images[index]
        pixelated_image = (pixelated_image / 255).astype(np.float32)
        known_array = self.known_arrays[index]

        full_image = np.concatenate((pixelated_image, known_array), axis=0)

        return full_image, known_array

    def __len__(self):
        return len(self.pixelated_images)
