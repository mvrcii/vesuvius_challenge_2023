import numpy as np
from skimage.transform import resize

from data_modules.abstract.abstract_dataset import AbstractDataset


class SegFormerDataset(AbstractDataset):
    def __init__(self, root_dir, images, label_size, patch_size, transform, labels=None):
        super().__init__(root_dir=root_dir,
                         images=images,
                         transform=transform,
                         labels=labels,
                         label_size=label_size,
                         patch_size=patch_size)
