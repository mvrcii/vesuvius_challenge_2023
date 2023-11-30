import os
import random

import albumentations as A
import numpy as np
from lightning.pytorch import LightningDataModule
from skimage.transform import resize
from torch.utils.data import Dataset, DataLoader
from transformers import SegformerImageProcessor

from data.create_dataset import build_dataset_dir_from_config


class SegFormerDataModule(LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.data_root_dir = build_dataset_dir_from_config(config=cfg)

    def train_dataloader(self):
        return self.build_dataloader(dataset_type='train')

    def val_dataloader(self):
        return self.build_dataloader(dataset_type='val')

    def get_transforms(self, dataset_type):
        image_processor = SegformerImageProcessor.from_pretrained(pretrained_model_name_or_path=self.cfg.from_pretrained)

        mean, std = image_processor.image_mean, image_processor.image_std

        mean.append(np.mean(mean))
        mean = np.array(mean)
        assert len(mean) == self.cfg.in_chans

        std.append(np.mean(std))
        std = np.array(std)
        assert len(std) == self.cfg.in_chans

        normalize = A.Normalize(mean=mean, std=std)

        transforms = []
        if dataset_type == 'train':
            transforms = self.cfg.train_aug
        elif dataset_type == 'val':
            transforms = self.cfg.val_aug

        transforms.append(normalize)

        return A.Compose(transforms=transforms, is_check_shapes=False)

    def build_dataloader(self, dataset_type):
        data_dir = os.path.join(self.data_root_dir, dataset_type)

        img_dir = os.path.join(data_dir, 'images')
        label_dir = os.path.join(data_dir, 'labels')

        images_list = os.listdir(img_dir)
        label_list = os.listdir(label_dir)

        if not images_list or not label_list:
            raise NotADirectoryError(f"One or both directories (images, labels) of {data_dir} are empty.")

        num_samples = int(len(images_list) * self.cfg.dataset_fraction)

        images_list.sort()
        label_list.sort()

        images_list = images_list[:num_samples]
        label_list = label_list[:num_samples]

        transform = self.get_transforms(dataset_type=dataset_type)

        dataset = WuesuvDataset(img_dir=img_dir,
                                label_dir=label_dir,
                                images=images_list,
                                labels=label_list,
                                label_size=self.cfg.label_size,
                                transform=transform)

        data_loader = DataLoader(dataset,
                                 batch_size=self.cfg.train_batch_size if dataset_type == 'train' else self.cfg.val_batch_size,
                                 shuffle=(dataset_type == 'train'),
                                 num_workers=self.cfg.num_workers,
                                 pin_memory=True,
                                 drop_last=False,
                                 persistent_workers=True)

        return data_loader


class WuesuvDataset(Dataset):
    def __init__(self, img_dir, label_dir, images, label_size, transform, labels=None):
        self.images = np.array(images)
        self.labels = labels
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform
        self.label_shape = (label_size, label_size)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = np.load(os.path.join(self.img_dir, self.images[idx]), allow_pickle=True)
        label = np.load(os.path.join(self.label_dir, self.labels[idx]), allow_pickle=True)

        if label.dtype == object:
            label = label.astype(np.uint8)

        # Add random augmentation on the layer axis
        if random.random() < 0.5:
            np.random.shuffle(image)

        # Rearrange image from (channels, height, width) to (height, width, channels) to work with albumentations
        image = np.transpose(image, (1, 2, 0))

        # Apply augmentations and normalization
        if self.transform:
            augmented = self.transform(image=image, mask=label)
            image, label = augmented['image'], augmented['mask']

        # Rearrange image back from (height, width, channels) to (channels, height, width) to work with segformer input
        image = np.transpose(image, (2, 0, 1))

        # Scale down label to match segformer output
        label = resize(label, self.label_shape, order=0, preserve_range=True, anti_aliasing=False)

        return image, label
