import os
from random import sample

import numpy as np
from lightning.pytorch import LightningDataModule
from torch.utils.data import Dataset, DataLoader


class EfficientNetV2DataModule(LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        dataset_name = f'slice'
        self.data_root_dir = os.path.join(cfg.dataset_target_dir, dataset_name)

    def train_dataloader(self):
        """Returns the training data loader."""
        return self.build_dataloader(dataset_type='train')

    def val_dataloader(self):
        """Returns the validation data loader."""
        return self.build_dataloader(dataset_type='val')

    def build_dataloader(self, dataset_type):
        img_dir = os.path.join(self.data_root_dir, dataset_type, 'images')

        norm_params = np.load(os.path.join(self.data_root_dir, 'train', "norm_params.npz"))
        mean = norm_params['mean']
        std = norm_params['std']

        images = os.listdir(img_dir)
        num_images = int(len(images) * self.cfg.dataset_fraction)
        images = sample(images, num_images)

        dataset = SliceDataset(img_dir=img_dir, images=images, mean=mean, std=std)


        data_loader = DataLoader(dataset,
                                 batch_size=self.cfg.train_batch_size if dataset_type == 'train' else self.cfg.val_batch_size,
                                 shuffle=(dataset_type == 'train'),
                                 num_workers=self.cfg.num_workers,
                                 pin_memory=True,
                                 drop_last=False,
                                 persistent_workers=True)

        return data_loader


class SliceDataset(Dataset):
    def __init__(self, img_dir, images, mean, std):
        self.images = images
        self.img_dir = img_dir
        self.mean = mean
        self.std = std

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = np.load(os.path.join(self.img_dir, self.images[idx]))

        # Expand dims to create a single-channel image
        image = np.expand_dims(image, 0)

        self.mean = 0.5
        self.std = 0.5

        if self.mean is None or self.std is None:
            raise RuntimeError("Mean and standard deviation have to be provided for normalization.")

        image = (image - self.mean) / self.std

        label = float(int(self.images[idx].split('.')[0].split('_')[1]) / 100)

        return image, label
