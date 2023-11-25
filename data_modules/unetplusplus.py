import os

import numpy as np
from lightning.pytorch import LightningDataModule
from skimage.transform import resize
from torch.utils.data import Dataset, DataLoader


class UnetPlusPlusDataModule(LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        dataset_name = f'single_fold_{cfg.patch_size}px'
        self.data_root_dir = os.path.join(cfg.dataset_target_dir, dataset_name)

    def train_dataloader(self):
        """Returns the training data loader."""
        return self.build_dataloader(dataset_type='train')

    def val_dataloader(self):
        """Returns the validation data loader."""
        return self.build_dataloader(dataset_type='val')

    def build_dataloader(self, dataset_type):
        data_dir = os.path.join(self.data_root_dir, dataset_type)

        img_dir = os.path.join(data_dir, 'images')
        label_dir = os.path.join(data_dir, 'labels')

        assert os.path.isdir(img_dir)
        assert os.path.isdir(label_dir)

        images_list = os.listdir(img_dir)
        label_list = os.listdir(label_dir)

        if not images_list or not label_list:
            raise ValueError(f"One or both directories (images, labels) of {data_dir} are empty.")

        num_samples = int(len(images_list) * self.cfg.dataset_fraction)

        images_list.sort()
        label_list.sort()

        images_list = images_list[:num_samples]
        label_list = label_list[:num_samples]

        dataset = WuesuvDataset(
            img_dir=img_dir,
            label_dir=label_dir,
            images=images_list,
            labels=label_list
        )

        batch_size = self.cfg.train_batch_size if dataset_type == 'train' else self.cfg.val_batch_size
        data_loader = DataLoader(dataset,
                                 batch_size=batch_size,
                                 shuffle=(dataset_type == 'train'),
                                 num_workers=self.cfg.num_workers,
                                 pin_memory=True,
                                 drop_last=False,
                                 persistent_workers=True)

        return data_loader


class WuesuvDataset(Dataset):
    def __init__(self, img_dir, label_dir, images, labels=None):
        self.images = np.array(images)
        self.labels = labels
        self.img_dir = img_dir
        self.label_dir = label_dir

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = np.load(os.path.join(self.img_dir, self.images[idx]))
        label = np.load(os.path.join(self.label_dir, self.labels[idx]))

        # Rearrange image from (channels, height, width) to (height, width, channels) to work with albumentations
        #image = np.transpose(image, (1, 2, 0))

        # (H,W) -> (1,H,W)
        label = np.expand_dims(label, axis=0)

        assert image.ndim == 3 and label.ndim == 3

        return image, label
