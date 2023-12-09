import os

import numpy as np
from torch.utils.data import Dataset
from skimage.transform import resize


class AbstractDataset(Dataset):
    def __init__(self, root_dir, images, label_size, patch_size, transform, labels=None):
        self.images = np.array(images)
        self.labels = labels
        self.root_dir = root_dir
        self.transform = transform
        self.patch_shape = (patch_size, patch_size)
        self.label_shape = (label_size, label_size)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx, label_shape=None, patch_shape=None):
        image = np.load(os.path.join(self.root_dir, self.images[idx]))
        label = np.load(os.path.join(self.root_dir, self.labels[idx]))
        label = np.unpackbits(label).reshape(label_shape)

        # Scale label up to patch shape
        label = resize(label, patch_shape, order=0, preserve_range=True, anti_aliasing=False)

        # Rearrange image from (channels, height, width) to (height, width, channels) to work with albumentations
        image = np.transpose(image, (1, 2, 0))

        # Apply augmentations and normalization
        if self.transform:
            augmented = self.transform(image=image, mask=label)
            image, label = augmented['image'], augmented['mask']

        # Rearrange image back from (height, width, channels) to (channels, height, width) to work with segformer input
        image = np.transpose(image, (2, 0, 1))

        # Scale label back down to label shape
        label = resize(label, label_shape, order=0, preserve_range=True, anti_aliasing=False)

        return image, label
