import numpy as np
from skimage.transform import resize

from data_modules.abstract.abstract_dataset import AbstractDataset


class SegFormerDataset(AbstractDataset):
    def __init__(self, root_dir, images, label_size, patch_size, transform, labels=None):
        super().__init__(root_dir, images, transform, labels, label_size, patch_size)
