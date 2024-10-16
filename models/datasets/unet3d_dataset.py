import os
import re

import numpy as np
import torch
from torch import float16
from torch.utils.data import Dataset


class UNET3D_Dataset(Dataset):
    def __init__(self, cfg, root_dir, images, transform, labels=None):
        self.cfg = cfg
        self.images = np.array(images)
        self.labels = labels
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = np.load(os.path.join(self.root_dir, self.images[idx]))

        match = re.search(r'l(\d+)', self.images[idx].split('_')[-1])
        if match:
            number = int(match.group(1)) / 100.0
        else:
            number = 0.0
        label = torch.tensor(number, dtype=torch.float16)

        # Rearrange image from (channels, height, width) to (height, width, channels) to work with albumentations
        image = np.transpose(image, (1, 2, 0))

        # Apply augmentations and normalization
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']

        # Rearrange image back from (height, width, channels) to (channels, height, width) to work with segformer input
        image = np.transpose(image, (2, 0, 1))

        image = torch.tensor(image, dtype=float16).unsqueeze(0)

        return image, label
