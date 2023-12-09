import numpy as np
from skimage.transform import resize

from data_modules.abstract.abstract_dataset import AbstractDataset


class SegFormerDataset(AbstractDataset):
    def __init__(self, root_dir, images, label_size, transform, labels=None):
        super().__init__(root_dir, images, transform, labels)
        self.label_shape = (label_size, label_size)

    def __getitem__(self, idx):
        image, label = super().__getitem__(idx)

        # unpack label from binary data and reshape it to original shape
        label = np.unpackbits(label).reshape(self.label_shape)

        # resize label to fit segformer output
        # not necessary, done in dataset creation
        # label = resize(label, self.label_shape, order=0, preserve_range=True, anti_aliasing=False)
        return image, label
