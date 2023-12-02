import os
import sys

import albumentations as A
import numpy as np
import pandas as pd
from lightning.pytorch import LightningDataModule
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from transformers import SegformerImageProcessor

from constants import get_frag_name_from_id


def generate_random_balanced_dataset(seed, csv_path, dataset_fraction, frag_ids, ink_threshold, artefact_threshold, train_split):
    try:
        data = pd.read_csv(csv_path)
    except Exception as e:
        print(e)
        sys.exit(1)

    if seed == -1:
        seed = None  # Set random seed if -1 is given

    # Filter for desired fragments
    data = data[data['frag_id'].isin(frag_ids)]

    # Select ink samples
    ink_samples = data[data['ink_p'] > ink_threshold]

    # Calculate the number of non-ink samples to select
    num_ink_samples = len(ink_samples)

    # Split non-ink samples into two groups
    non_ink_samples_no_artefact = data[(data['ink_p'] <= ink_threshold) & (data['artefact_p'] == 0)]
    non_ink_samples_with_artefact = data[(data['ink_p'] <= ink_threshold) & (data['artefact_p'] >= artefact_threshold)]

    # Determine the available number of non-ink samples with artefacts
    available_with_artefact = len(non_ink_samples_with_artefact)
    desired_with_artefact = int(num_ink_samples * 0.3)

    # Adjust the number of each non-ink group based on availability
    num_with_artefact_samples = min(desired_with_artefact, available_with_artefact)
    num_no_artefact_samples = num_ink_samples - num_with_artefact_samples

    # Ensure not to exceed the available non-ink samples with no artefacts
    num_no_artefact_samples = min(num_no_artefact_samples, len(non_ink_samples_no_artefact))

    # Select samples from each non-ink group
    selected_no_artefact_samples = non_ink_samples_no_artefact.sample(n=num_no_artefact_samples, random_state=seed)
    selected_with_artefact_samples = non_ink_samples_with_artefact.sample(n=num_with_artefact_samples, random_state=seed)

    # Combine all selected samples
    balanced_dataset = pd.concat([ink_samples, selected_no_artefact_samples, selected_with_artefact_samples])

    # Print statistics
    print(f"Total ink samples: {num_ink_samples}")
    print(f"Total non-ink samples with no artefact: {num_no_artefact_samples}")
    print(f"Total non-ink samples with artefact > {artefact_threshold}: {num_with_artefact_samples}")

    balanced_dataset['file_path'] = balanced_dataset.apply(
        lambda row: os.path.join(get_frag_name_from_id(row['frag_id']), 'images', row['filename']), axis=1)

    train_df, valid_df = train_test_split(balanced_dataset, train_size=train_split, random_state=seed)

    if dataset_fraction != 1.0 or dataset_fraction != 1:
        train_df, _ = train_test_split(train_df,
                                       train_size=round(len(train_df.index) * dataset_fraction),
                                       random_state=seed)
        valid_df, _ = train_test_split(valid_df,
                                       train_size=round(len(valid_df.index) * dataset_fraction),
                                       random_state=seed)

    train_image_paths = train_df['file_path'].tolist()
    val_image_paths = valid_df['file_path'].tolist()

    train_label_paths = [path.replace('images', 'labels') for path in train_image_paths]
    val_label_paths = [path.replace('images', 'labels') for path in train_image_paths]

    return train_image_paths, train_label_paths, val_image_paths, val_label_paths


class SegFormerDataModule(LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        csv_path = os.path.join(cfg.dataset_target_dir, str(cfg.patch_size), 'label_infos.csv')
        self.train_image_paths, self.train_label_paths, self.val_image_paths, self.val_label_paths = generate_random_balanced_dataset(
            seed=cfg.seed,
            csv_path=csv_path,
            dataset_fraction=cfg.dataset_fraction,
            frag_ids=cfg.fragment_ids,
            ink_threshold=cfg.ink_ratio,
            artefact_threshold=cfg.artefact_ratio,
            train_split=cfg.train_split)

    def train_dataloader(self):
        return self.build_dataloader(dataset_type='train')

    def val_dataloader(self):
        return self.build_dataloader(dataset_type='val')

    def get_transforms(self, dataset_type):
        image_processor = SegformerImageProcessor.from_pretrained(
            pretrained_model_name_or_path=self.cfg.from_pretrained)

        mean, std = image_processor.image_mean, image_processor.image_std

        mean.append(np.mean(mean))
        mean = np.array(mean)
        assert len(mean) == self.cfg.in_chans

        std.append(np.mean(std))
        std = np.array(std)
        assert len(std) == self.cfg.in_chans

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
            images_list = self.train_image_paths
            label_list = self.train_label_paths
        else:
            images_list = self.val_image_paths
            label_list = self.val_label_paths

        transform = self.get_transforms(dataset_type=dataset_type)
        root_dir = os.path.join(self.cfg.dataset_target_dir, str(self.cfg.patch_size))
        dataset = WuesuvDataset(root_dir=root_dir,
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
    def __init__(self, root_dir, images, label_size, transform, labels=None):
        self.images = np.array(images)
        self.labels = labels
        self.root_dir = root_dir
        self.transform = transform
        self.label_shape = (label_size, label_size)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = np.load(os.path.join(self.root_dir, self.images[idx]))
        label = np.load(os.path.join(self.root_dir, self.labels[idx]))

        # Add random augmentation on the layer axis
        # if random.random() < 0.5:
        #     np.random.shuffle(image)

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
