import sys

import pandas as pd

from sklearn.model_selection import train_test_split
from models.data_modules.abstract_datamodule import AbstractDataModule
import os
from models.datasets.unet3dsf_dataset import UNET3D_SFDataset
from utility.configs import Config


class UNET3D_SFDataModule(AbstractDataModule):
    def __init__(self, cfg):
        super().__init__(cfg=cfg)

    def get_dataset_class(self):
        return UNET3D_SFDataset

    def generate_dataset(self, cfg: Config):
        csv_path = os.path.join(cfg.dataset_target_dir, str(cfg.patch_size), 'label_infos.csv')

        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            print(e)
            sys.exit(1)

        if cfg.seed == -1:
            cfg.seed = None  # Set random seed if -1 is given

        if getattr(cfg, "max_ignore_th", False) and "ignore_p" in df.columns:
            print("Before ignoring: ", len(df.index))
            df = df[df["ignore_p"] < cfg.max_ignore_th]
            print(f"After ignoring patches with ignore_p > {cfg.max_ignore_th}: ", len(df.index))

        if not getattr(cfg, "take_full_dataset", False):
            count_zero = (df['ink_p'] == 0).sum()
            count_greater_than_zero = (df['ink_p'] > 0).sum()
            print(
                f"Before balancing: {count_zero} samples with ink_p = 0 and {count_greater_than_zero} samples with ink_p > 0")

            # BALANCING IS DONE ON CREATION
            # Step 1: Filter out rows where ink_p > ratio
            df_ink_p_greater_than_ink_ratio = df[df['ink_p'] >= cfg.ink_ratio]

            # Step 2: Decide how many no-ink samples
            no_ink_sample_count = int(len(df_ink_p_greater_than_ink_ratio) * cfg.no_ink_sample_percentage)

            # Step 3: Select the correct amount of samples with 0 ink
            df_good_no_inks = df[df['ink_p'] == 0].head(no_ink_sample_count)

            # # Step 4: Concatenate the two DataFrames
            df = pd.concat([df_ink_p_greater_than_ink_ratio, df_good_no_inks])

        count_zero = (df['ink_p'] == 0).sum()
        count_greater_than_zero = (df['ink_p'] > 0).sum()
        print(f"Final Count: {count_zero} samples with ink_p = 0 and {count_greater_than_zero} samples with ink_p > 0")

        # Preprocessing to correct file paths
        df['file_path'] = df.apply(self.generate_file_path, axis=1)

        validation_fragments_int = getattr(cfg, 'validation_fragments', None)

        validation_fragments = []
        for x in validation_fragments_int:
            validation_fragments.append(int(x))

        if validation_fragments is None or len(validation_fragments) == 0:
            raise Exception("Validation fragments not specified or empty!")

        train_df = df[~df['frag_id'].isin(validation_fragments)]
        valid_df = df[df['frag_id'].isin(validation_fragments)]

        if train_df.shape[0] == 0:
            raise Exception("Training DataFrame is empty. No entries found for the given validation IDs.")

        if valid_df.shape[0] == 0:
            raise Exception("Validation DataFrame is empty. No entries found for the given validation IDs.")

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
