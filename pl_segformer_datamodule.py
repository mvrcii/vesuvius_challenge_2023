import os
import sys

import albumentations as A
import numpy as np
from lightning.pytorch import LightningDataModule
from skimage.transform import resize
from torch.utils.data import Dataset, DataLoader


class SegFormerDataModule(LightningDataModule):
    """
    Data module for SegFormer using PyTorch Lightning.

    Args:
        cfg: Config.
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.data_root_dir = self._get_data_root_dir(cfg)

    @staticmethod
    def _get_data_root_dir(config):
        if config.dataset_in_chans != config.in_chans:
            print(f"Input channels of dataset {config.dataset_in_chans} != Input channels of model {config.in_chans}")
            sys.exit(1)

        if config.k_fold:
            print("Start training with k_fold data")
            dataset_name = f'k_fold_{config.patch_size}px_{config.dataset_in_chans}ch'
            data_root_dir = os.path.join(config.dataset_target_dir, dataset_name)
        else:
            print("Start training with single fragment data")
            dataset_name = f'single_fold_{config.patch_size}px_{config.dataset_in_chans}ch'
            data_root_dir = os.path.join(config.dataset_target_dir, dataset_name)

        return data_root_dir

    def _build_data_loader(self, dataset_type: str) -> DataLoader:
        """
        Build and return a data loader for the specified dataset type.

        Args:
            dataset_type (str): Type of the dataset ('train' or 'val').

        Returns:
            DataLoader: The data loader for the specified dataset type.
        """
        try:
            return build_dataloader(data_root_dir=self.data_root_dir, cfg=self.cfg, dataset_type=dataset_type)
        except DataLoaderError as e:
            print(f"\nCritical error in data loading: {e}")
            sys.exit(1)

    def train_dataloader(self) -> DataLoader:
        """Returns the training data loader."""
        return self._build_data_loader(dataset_type='train')

    def val_dataloader(self) -> DataLoader:
        """Returns the validation data loader."""
        return self._build_data_loader(dataset_type='val')


class DataLoaderError(Exception):
    """Custom exception class for data loader errors."""
    pass


def build_dataloader(data_root_dir, cfg, dataset_type='train'):
    data_dir = os.path.join(data_root_dir, dataset_type)

    img_dir = os.path.join(data_dir, 'images')
    label_dir = os.path.join(data_dir, 'labels')

    try:
        images_list = os.listdir(img_dir)
    except FileNotFoundError:
        raise DataLoaderError(f"Image directory not found: {img_dir}")

    try:
        label_list = os.listdir(label_dir)
    except FileNotFoundError:
        raise DataLoaderError(f"Label directory not found: {label_dir}")

    if not images_list or not label_list:
        raise DataLoaderError(f"One or both directories (images, labels) of {data_dir} are empty.")

    num_samples = int(len(images_list) * cfg.dataset_fraction)

    images_list.sort()
    label_list.sort()

    images_list = images_list[:num_samples]
    label_list = label_list[:num_samples]

    # Get transformations
    common_aug, image_aug = get_transforms(dataset_type, cfg)

    dataset = WuesuvDataset(data_dir=data_dir,
                            img_dir=img_dir,
                            label_dir=label_dir,
                            images=images_list,
                            labels=label_list,
                            in_chans=cfg.in_chans,
                            common_aug=common_aug,
                            image_aug=image_aug,
                            train=(dataset_type == 'train'))

    data_loader = DataLoader(dataset,
                             batch_size=cfg.train_batch_size if dataset_type == 'train' else cfg.val_batch_size,
                             shuffle=(dataset_type == 'train'),
                             num_workers=cfg.num_workers,
                             pin_memory=True,
                             drop_last=False,
                             persistent_workers=True)

    return data_loader


def get_transforms(data, cfg):
    if data == 'train':
        return A.Compose(cfg.train_common_aug, is_check_shapes=False), A.Compose(cfg.train_image_aug,
                                                                                 is_check_shapes=False)
    elif data == 'val':
        return A.Compose(cfg.val_common_aug, is_check_shapes=False), A.Compose(cfg.val_image_aug, is_check_shapes=False)


class WuesuvDataset(Dataset):
    def __init__(self, data_dir, img_dir, label_dir, images, in_chans, labels=None, common_aug=None, image_aug=None,
                 train=True):
        self.images = np.array(images)
        self.labels = labels
        self.data_dir = data_dir
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.common_aug = common_aug
        self.image_aug = image_aug
        self.train = train
        self.in_chans = in_chans

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = np.load(os.path.join(self.img_dir, self.images[idx]))
        label = np.load(os.path.join(self.label_dir, self.labels[idx]))

        # Rearrange image from (channels, height, width) to (height, width, channels) to work with albumentations
        image = np.transpose(image, (1, 2, 0))

        # Get original dimensions
        patch_size = image.shape[0]
        label_size = label.shape[0]

        # Scale up label to match image size
        if label_size != patch_size:
            label = resize(label, (patch_size, patch_size), order=0, preserve_range=True, anti_aliasing=False)

        # Apply common augmentations to both image and label
        if self.common_aug:
            augmented = self.common_aug(image=image, mask=label)
            image, label = augmented['image'], augmented['mask']

        # Apply image-specific augmentations (like resizing)
        if self.image_aug:
            augmented = self.image_aug(image=image)
            image = augmented['image']

        # Rearrange image back to (channels, height, width) from (height, width, channels) to work with segformer input
        image = np.transpose(image, (2, 0, 1))

        # Scale down label to match segformer output
        if label_size != patch_size:
            label = resize(label, (label_size, label_size), order=0, preserve_range=True, anti_aliasing=False)

        return image, label

    def load_image_and_mask(self, idx):
        image = np.load(os.path.join(self.img_dir, self.images[idx]))
        label = np.load(os.path.join(self.label_dir, self.labels[idx]))
        return image, label
