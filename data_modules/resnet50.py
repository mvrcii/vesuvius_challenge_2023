import os

import numpy as np
from lightning.pytorch import LightningDataModule
from torch.utils.data import Dataset, DataLoader


class ResNet50DataModule(LightningDataModule):
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
        data_dir = os.path.join(self.data_root_dir, dataset_type)
        img_dir = os.path.join(data_dir, 'images')

        assert os.path.isdir(img_dir)

        images_list = os.listdir(img_dir)

        if not images_list:
            raise ValueError(f"One or both directories (images) of {data_dir} are empty.")

        images_list.sort()

        dataset = SliceDataset(
            img_dir=img_dir,
            images=images_list,
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


class SliceDataset(Dataset):
    def __init__(self, img_dir, images):
        self.images = np.array(images)
        self.img_dir = img_dir

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = np.load(os.path.join(self.img_dir, self.images[idx]))

        label = float(int(self.images[idx].split('.')[0].split('_')[1]) / 100)

        assert image.ndim == 2

        return image, label
