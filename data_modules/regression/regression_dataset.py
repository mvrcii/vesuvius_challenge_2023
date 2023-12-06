import random

import numpy as np

from data_modules.abstract.abstract_dataset import AbstractDataset


def layer_shift_and_dropout(image):
    image_tmp = np.zeros_like(image)
    cropping_num = random.randint(22, 30)

    in_chans = image.shape[0]
    start_idx = random.randint(0, in_chans - cropping_num)
    crop_indices = np.arange(start_idx, start_idx + cropping_num)

    start_paste_idx = random.randint(0, in_chans - cropping_num)

    tmp = np.arange(start_paste_idx, cropping_num)
    np.random.shuffle(tmp)

    cutout_idx = random.randint(0, 2)
    temporal_random_cutout_idx = tmp[:cutout_idx]

    image_tmp[..., start_paste_idx: start_paste_idx + cropping_num] = image[..., crop_indices]

    if random.random() > 0.4:
        image_tmp[..., temporal_random_cutout_idx] = 0
    image = image_tmp

    return image


class RegressionDataset(AbstractDataset):
    def __init__(self, root_dir, images, transform, labels=None):
        super().__init__(root_dir, images, transform, labels)

    def __getitem__(self, idx):
        image, label = super().__getitem__(idx)

        image = layer_shift_and_dropout(image)

        return image, label
