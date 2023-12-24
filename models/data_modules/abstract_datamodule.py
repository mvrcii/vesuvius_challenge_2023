import os
import sys

import albumentations as A
import pandas as pd
from lightning.pytorch import LightningDataModule
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from utility.config_handler import Config
from models.datasets.abstract_dataset import AbstractDataset
from utility.fragment import get_frag_name_from_id


class AbstractDataModule(LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.t_img_paths, self.t_label_paths, self.v_img_paths, self.v_label_paths = self.generate_dataset(cfg=cfg)
        self.dataset = self.get_dataset_class()

    @staticmethod
    def get_dataset_class():
        return AbstractDataset

    def train_dataloader(self):
        return self.build_dataloader(dataset_type='train')

    def val_dataloader(self):
        return self.build_dataloader(dataset_type='val')

    def get_transforms(self, dataset_type):
        if dataset_type == 'train':
            transforms = self.cfg.train_aug
            return A.Compose(transforms=transforms, is_check_shapes=False)
        elif dataset_type == 'val':
            transforms = self.cfg.val_aug
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
                               label_size=self.cfg.label_size,
                               patch_size=self.cfg.patch_size,
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

    def generate_dataset(self, cfg: Config):
        csv_path = os.path.join(cfg.dataset_target_dir, str(cfg.patch_size), 'label_infos.csv')

        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            print(e)
            sys.exit(1)

        if cfg.seed == -1:
            cfg.seed = None  # Set random seed if -1 is given

        if "ignore_p" in df.columns:
            print("Before ignoring: ", len(df.index))
            df = df[df["ignore_p"] < cfg.max_ignore_th]
            print(f"After ignoring patches with ignore_p > {cfg.max_ignore_th}: ", len(df.index))

        if not cfg.take_full_dataset:
            count_zero = (df['ink_p'] == 0).sum()
            count_greater_than_zero = (df['ink_p'] > 0).sum()
            print(
                f"Before balancing: {count_zero} samples with ink_p = 0 and {count_greater_than_zero} samples with ink_p > 0")

            # BALANCING IS DONE ON CREATION
            # Step 1: Filter out rows where ink_p > ratio
            df_ink_p_greater_than_ink_ratio = df[df['ink_p'] > cfg.ink_ratio]

            # Step 2: Decide how many no-ink samples
            no_ink_sample_count = int(len(df_ink_p_greater_than_ink_ratio) * cfg.no_ink_sample_percentage)

            # Step 3: Filter out rows where ink_p <= 0 and limit the number of rows
            df_good_no_inks = df[df['ink_p'] == 0].head(no_ink_sample_count)

            # # Step 4: Concatenate the two DataFrames
            df = pd.concat([df_ink_p_greater_than_ink_ratio, df_good_no_inks])

        count_zero = (df['ink_p'] == 0).sum()
        count_greater_than_zero = (df['ink_p'] > 0).sum()
        print(f"Final Count: {count_zero} samples with ink_p = 0 and {count_greater_than_zero} samples with ink_p > 0")

        df['file_path'] = df.apply(
            lambda row: os.path.join(get_frag_name_from_id(row['frag_id']), 'images', row['filename']), axis=1)

        train_df, valid_df = train_test_split(df, train_size=cfg.train_split, random_state=cfg.seed)

        if cfg.dataset_fraction != 1.0 or cfg.dataset_fraction != 1:
            train_df, _ = train_test_split(train_df,
                                           train_size=round(len(train_df.index) * cfg.dataset_fraction),
                                           random_state=cfg.seed)
            valid_df, _ = train_test_split(valid_df,
                                           train_size=round(len(valid_df.index) * cfg.dataset_fraction),
                                           random_state=cfg.seed)

        train_image_paths = train_df['file_path'].tolist()
        val_image_paths = valid_df['file_path'].tolist()

        train_label_paths = [path.replace('images', 'labels') for path in train_image_paths]
        val_label_paths = [path.replace('images', 'labels') for path in val_image_paths]

        print(f"Total train samples: {len(train_image_paths)}")
        print(f"Total validation samples: {len(val_image_paths)}")

        return train_image_paths, train_label_paths, val_image_paths, val_label_paths
