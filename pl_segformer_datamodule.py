import os

import albumentations as A
import numpy as np
from lightning.pytorch import LightningDataModule
from skimage.transform import resize
from torch.utils.data import Dataset, DataLoader

from conf import CFG


class SegFormerDataModule(LightningDataModule):
    def __init__(self):
        super().__init__()

    def train_dataloader(self):
        return build_dataloader(
            data_root_dir=os.path.join(CFG.data_root_dir, str(CFG.size)),
            dataset_type='train')

    def val_dataloader(self):
        return build_dataloader(
            data_root_dir=os.path.join(CFG.data_root_dir, str(CFG.size)),
            dataset_type='val')

    # Implement test_dataloader if needed


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

    # Get transformations
    common_aug, image_aug = get_transforms(dataset_type, CFG)

    dataset = WuesuvDataset(data_dir=data_dir,
                            img_dir=img_dir,
                            label_dir=label_dir,
                            images=images_list,
                            labels=label_list,
                            cfg=CFG,
                            common_aug=common_aug,
                            image_aug=image_aug,
                            train=(dataset_type == 'train'))

    data_loader = DataLoader(dataset,
                             batch_size=CFG.train_batch_size if dataset_type == 'train' else CFG.val_batch_size,
                             shuffle=(dataset_type == 'train'),
                             num_workers=CFG.num_workers,
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
    def __init__(self, data_dir, img_dir, label_dir, images, cfg, labels=None, common_aug=None, image_aug=None,
                 train=True):
        self.images = np.array(images)
        self.cfg = cfg
        self.labels = labels
        self.data_dir = data_dir
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.common_aug = common_aug
        self.image_aug = image_aug
        self.train = train

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = np.load(os.path.join(self.img_dir, self.images[idx]))
        label = np.load(os.path.join(self.label_dir, self.labels[idx]))

        # Rearrange image from (channels, height, width) to (height, width, channels) to work with albumentations
        image = np.transpose(image, (1, 2, 0))
        # Scale up label to match image size
        label = resize(label, (512, 512), order=0, preserve_range=True, anti_aliasing=False)

        # Apply common augmentations to both image and label
        if self.common_aug and CFG.use_aug:
            augmented = self.common_aug(image=image, mask=label)
            image, label = augmented['image'], augmented['mask']

        # Apply image-specific augmentations (like resizing)
        if self.image_aug and CFG.use_aug:
            augmented = self.image_aug(image=image)
            image = augmented['image']

        # if self.train and random.random() < 0.5:  # 50% prob to apply mixup/cutmix
        #     image2, label2 = self.load_image_and_mask(random.randint(0, self.__len__() - 1))
        #     if random.random() < 0.5:  # Another 50% prob to choose between mixup/cutmix
        #         if self.cfg.use_mixup:
        #             image, label = self.apply_mixup(image, label, image2, label2)
        #     else:
        #         if self.cfg.use_cutmix:
        #             image, label = self.apply_cutmix(image, label, image2, label2)

        # Rearrange image back to (channels, height, width) from (height, width, channels) to work with segformer input
        image = np.transpose(image, (2, 0, 1))
        # Scale down label to match segformer output
        label = resize(label, (128, 128), order=0, preserve_range=True, anti_aliasing=False)
        return image, label

    @staticmethod
    def apply_mixup(image1, label1, image2, label2):
        # Generate mixing factor from Beta distribution
        alpha = 0.2  # This can be a hyperparameter
        lam = np.random.beta(alpha, alpha)
        mixed_image = lam * image1 + (1 - lam) * image2
        mixed_label = lam * label1 + (1 - lam) * label2
        return mixed_image, mixed_label

    @staticmethod
    def apply_cutmix(image1, label1, image2, label2):
        # Randomly choose the region
        h, w = image1.shape[0], image1.shape[1]
        cx, cy = np.random.randint(w), np.random.randint(h)
        w2, h2 = np.random.randint(w // 2), np.random.randint(h // 2)
        x1, x2 = np.clip(cx - w2 // 2, 0, w), np.clip(cx + w2 // 2, 0, w)
        y1, y2 = np.clip(cy - h2 // 2, 0, h), np.clip(cy + h2 // 2, 0, h)

        mixed_image = np.array(image1)
        mixed_label = np.array(label1)
        mixed_image[y1:y2, x1:x2] = image2[y1:y2, x1:x2]
        mixed_label[y1:y2, x1:x2] = label2[y1:y2, x1:x2]

        return mixed_image, mixed_label

    def load_image_and_mask(self, idx):
        image = np.load(os.path.join(self.img_dir, self.images[idx]))
        label = np.load(os.path.join(self.label_dir, self.labels[idx]))
        return image, label
