import os

import albumentations as A
import numpy as np
from torch.utils.data import Dataset, DataLoader

from conf import CFG


def get_transforms(data, cfg):
    if data == 'train':
        aug = A.Compose(cfg.train_aug_list)
    elif data == 'val':
        aug = A.Compose(cfg.valid_aug_list)
    return aug


class CustomDataset(Dataset):
    def __init__(self, data_dir, img_dir, label_dir, images, cfg, labels=None, transform=None):
        self.images = np.array(images)
        self.cfg = cfg
        self.labels = labels
        self.transform = transform
        self.data_dir = data_dir
        self.img_dir = img_dir
        self.label_dir = label_dir

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = np.load(os.path.join(self.img_dir, self.images[idx]))
        data = self.transform(image=image)
        image = data['image']

        label = np.load(os.path.join(self.label_dir, self.labels[idx]))
        if np.max(label) == 255:
            label = label // 255

        return image[None, :, :, :], label


def build_dataloader(data_root_dir, dataset_type='train'):
    data_dir = os.path.join(data_root_dir, dataset_type)

    img_dir = os.path.join(data_dir, 'images')
    label_dir = os.path.join(data_dir, 'labels')

    images_list = os.listdir(img_dir)
    label_list = os.listdir(label_dir)

    num_samples = int(len(images_list) * CFG.dataset_fraction)

    images_list.sort()
    label_list.sort()

    images_list = images_list[:num_samples]
    label_list = label_list[:num_samples]

    dataset = CustomDataset(data_dir=data_dir,
                            img_dir=img_dir,
                            label_dir=label_dir,
                            images=images_list,
                            labels=label_list,
                            cfg=CFG,
                            transform=get_transforms(data=dataset_type, cfg=CFG))

    data_loader = DataLoader(dataset,
                             batch_size=CFG.train_batch_size if dataset_type == 'train' else CFG.val_batch_size,
                             shuffle=True,
                             num_workers=CFG.num_workers,
                             pin_memory=True,
                             drop_last=False)

    return data_loader
