import os

import albumentations as A
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader

from data_modules.abstract.abstract_dataset import AbstractDataset
from data_modules.utils import generate_dataset


class AbstractDataModule(LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.t_img_paths, self.t_label_paths, self.v_img_paths, self.v_label_paths = generate_dataset(cfg=cfg)
        self.dataset = self.get_dataset_class()

    @staticmethod
    def get_dataset_class():
        return AbstractDataset

    def train_dataloader(self):
        return self.build_dataloader(dataset_type='train')

    def val_dataloader(self):
        return self.build_dataloader(dataset_type='val')

    def get_transforms(self, dataset_type):
        normalize = A.Normalize(mean=[0], std=[1])

        transforms = []
        if dataset_type == 'train':
            transforms = self.cfg.train_aug
        elif dataset_type == 'val':
            transforms = self.cfg.val_aug

        transforms.append(normalize)

        return A.Compose(transforms=transforms, is_check_shapes=False)

    def build_dataloader(self, dataset_type):
        if dataset_type == 'train':
            images_list = self.t_img_paths
            label_list = self.t_label_paths
        else:
            images_list = self.v_img_paths
            label_list = self.v_label_paths

        transform = self.get_transforms(dataset_type=dataset_type)
        root_dir = os.path.join(self.cfg.dataset_target_dir, str(self.cfg.patch_size))
        dataset = self.dataset(root_dir=root_dir,
                               images=images_list,
                               labels=label_list,
                               transform=transform)

        batch_size = self.cfg.train_batch_size if dataset_type == 'train' else self.cfg.val_batch_size
        shuffle = dataset_type == 'train'
        data_loader = DataLoader(dataset,
                                 batch_size=batch_size,
                                 shuffle=shuffle,
                                 num_workers=self.cfg.num_workers,
                                 pin_memory=True,
                                 drop_last=False,
                                 persistent_workers=True)

        return data_loader
