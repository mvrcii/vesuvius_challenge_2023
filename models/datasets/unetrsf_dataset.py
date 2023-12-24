import torch

from models.datasets.unet3dsf_dataset import UNET3D_SFDataset


class UNETR_SFDataset(UNET3D_SFDataset):
    def __init__(self, root_dir, images, transform, cfg, labels=None):
        super().__init__(cfg, root_dir, images, transform, labels)

    def __getitem__(self, idx):
        image, label = self.__getitem__(idx)

        # pad image to have 16 layers
        image = torch.cat([image, torch.zeros(1, 16 - image.shape[1], self.patch_size, self.patch_size)], dim=1)

        return image, label
