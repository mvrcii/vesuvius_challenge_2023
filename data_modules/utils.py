import os
import sys

import pandas as pd
from sklearn.model_selection import train_test_split

from config_handler import Config
from constants import get_frag_name_from_id


def generate_dataset(cfg: Config):
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
