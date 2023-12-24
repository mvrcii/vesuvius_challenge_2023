import os

import numpy as np
from skimage.transform import resize

from models.datasets.abstract_dataset import AbstractDataset


class SegFormerDataset(AbstractDataset):
    def __init__(self, cfg, root_dir, images, transform, labels=None):
        super().__init__(cfg, root_dir, images, transform, labels)
        self.patch_shape = (cfg.patch_size, cfg.patch_size)
        self.label_shape = (cfg.label_size, cfg.label_size)

    def __getitem__(self, idx):
        image = np.load(os.path.join(self.root_dir, self.images[idx]))
        label = np.load(os.path.join(self.root_dir, self.labels[idx]))
        label = np.unpackbits(label).reshape(self.label_shape)

        # Scale label up to patch shape
        label = resize(label, self.patch_shape, order=0, preserve_range=True, anti_aliasing=False)

        # Rearrange image from (channels, height, width) to (height, width, channels) to work with albumentations
        image = np.transpose(image, (1, 2, 0))

        # Apply augmentations and normalization
        if self.transform:
            augmented = self.transform(image=image, mask=label)
            image, label = augmented['image'], augmented['mask']

        # Rearrange image back from (height, width, channels) to (channels, height, width) to work with segformer input
        image = np.transpose(image, (2, 0, 1))

        # Scale label back down to label shape
        label = resize(label, self.label_shape, order=0, preserve_range=True, anti_aliasing=False)

        return image, label
