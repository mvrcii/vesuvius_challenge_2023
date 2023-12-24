import os

import numpy as np
from torch.utils.data import Dataset


class AbstractDataset(Dataset):
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
        label = np.load(os.path.join(self.root_dir, self.labels[idx]))

        # Rearrange image from (channels, height, width) to (height, width, channels) to work with albumentations
        image = np.transpose(image, (1, 2, 0))

        # Apply augmentations and normalization
        if self.transform:
            augmented = self.transform(image=image, mask=label)
            image, label = augmented['image'], augmented['mask']

        # Rearrange image back from (height, width, channels) to (channels, height, width) to work with segformer input
        image = np.transpose(image, (2, 0, 1))

        return image, label
