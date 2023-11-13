import os

import albumentations as A
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from conf import CFG


def get_transforms(data, cfg):
    if data == 'train':
        aug = A.Compose(cfg.train_aug_list)
    elif data == 'val':
        aug = A.Compose(cfg.valid_aug_list)
    return aug


class WuesuvDataset(Dataset):
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
        label = np.load(os.path.join(self.label_dir, self.labels[idx]))
        if np.max(label) == 255:
            label = label // 255

        original_label = label.copy()
        original_images = image.copy()

        if self.transform:
            transformed = self.transform(image=image, mask=label)
            image = transformed['image'].numpy()
            label = transformed['mask'].numpy()

        fig, ax = plt.subplots(2, 2, figsize=(10, 5))
        ax[0, 0].imshow(original_label)
        ax[0, 0].set_title("Original Label")

        ax[0, 1].imshow(label)
        ax[0, 1].set_title("Transformed Label")

        ax[1, 0].imshow(original_images[:, :, 0])
        ax[1, 0].set_title("Original Image")

        ax[1, 1].imshow(image[0])
        ax[1, 1].set_title("Transformed Image")

        plt.show()

        torch.tensor(image)
        torch.tensor(label)

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

    dataset = WuesuvDataset(data_dir=data_dir,
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
