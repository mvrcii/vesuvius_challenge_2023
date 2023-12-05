import numpy as np

from data_modules.abstract.abstract_dataset import AbstractDataset


class UnetPlusPlusDataset(AbstractDataset):
    def __init__(self, root_dir, images, transform, labels=None):
        super().__init__(root_dir, images, transform, labels)

    def __getitem__(self, idx):
        image, label = super().__getitem__(idx)

        # (H,W) -> (1,H,W)
        label = np.expand_dims(label, axis=0)
        assert image.ndim == 3 and label.ndim == 3

        return image, label
