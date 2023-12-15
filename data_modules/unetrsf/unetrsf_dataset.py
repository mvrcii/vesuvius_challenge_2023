import os

import numpy as np
import torch
from skimage.transform import resize

from data_modules.abstract.abstract_dataset import AbstractDataset


class UNETR_SFDataset(AbstractDataset):
    def __init__(self, root_dir, images, label_size, patch_size, transform, labels=None):
        super().__init__(root_dir=root_dir,
                         images=images,
                         transform=transform,
                         labels=labels,
                         label_size=label_size,
                         patch_size=patch_size)
        # label[0] is ink, label[1] is ignore
        self.label_shape = (2, label_size, label_size)
        self.label_shape_upscaled = (2, patch_size, patch_size)

    def __getitem__(self, idx):
        image = np.load(os.path.join(self.root_dir, self.images[idx]))
        label = np.load(os.path.join(self.root_dir, self.labels[idx]))
        label = np.unpackbits(label).reshape(self.label_shape)

        # Scale label up to patch shape
        label = resize(label, self.label_shape_upscaled, order=0, preserve_range=True, anti_aliasing=False)

        # Rearrange image from (channels, height, width) to (height, width, channels) to work with albumentations
        image = np.transpose(image, (1, 2, 0))

        # Also rearrange label from (2, height, width) to (height, width, 2) (label[0] label, label[1] ignore)
        label = np.transpose(label, (1, 2, 0))

        # Apply augmentations and normalization
        if self.transform:
            augmented = self.transform(image=image, mask=label)
            image, label = augmented['image'], augmented['mask']

        # Rearrange image back from (height, width, channels) to (channels, height, width) to work with segformer input
        image = np.transpose(image, (2, 0, 1))
        label = np.transpose(label, (2, 0, 1))

        # Scale label back down to label shape
        label = resize(label, self.label_shape, order=0, preserve_range=True, anti_aliasing=False)
        label = torch.tensor(label, dtype=torch.float16)

        # go from (layers, patch_size, patch_size) to (1, layers, patch_size, patch_size) => add "1" channel
        image = torch.tensor(image).unsqueeze(0)

        # pad image to have 16 layers
        image = torch.cat([image, torch.zeros(1, 16 - image.shape[1], 256, 256)], dim=1)
        # x = torch.cat([x, torch.zeros(1, 1, 4, 256, 256)], dim=2)

        return image, label
