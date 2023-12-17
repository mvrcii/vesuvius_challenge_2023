import os
import sys

import pandas as pd
from sklearn.model_selection import train_test_split

from config_handler import Config
from constants import get_frag_name_from_id


def balance_dataset(cfg: Config, data):
    # Filter for desired fragments
    data = data[data['frag_id'].isin(cfg.fragment_ids)]

    # Select ink samples
    ink_samples = data[data['ink_p'] > cfg.ink_ratio]

    # Calculate the number of non-ink samples to select
    num_ink_samples = len(ink_samples)

    # Split non-ink samples into two groups
    non_ink_samples_no_artefact = data[(data['ink_p'] <= cfg.ink_ratio) & (data['artefact_p'] == 0)]
    non_ink_samples_with_artefact = data[
        (data['ink_p'] <= cfg.ink_ratio) & (data['artefact_p'] >= cfg.artefact_threshold)]

    # Determine the available number of non-ink samples with artefacts
    available_with_artefact = len(non_ink_samples_with_artefact)
    desired_with_artefact = int(num_ink_samples * 0.3)

    # Adjust the number of each non-ink group based on availability
    num_with_artefact_samples = min(desired_with_artefact, available_with_artefact)
    num_no_artefact_samples = num_ink_samples - num_with_artefact_samples

    # Ensure not to exceed the available non-ink samples with no artefacts
    num_no_artefact_samples = min(num_no_artefact_samples, len(non_ink_samples_no_artefact))

    # Select samples from each non-ink group
    selected_no_artefact_samples = non_ink_samples_no_artefact.sample(n=num_no_artefact_samples, random_state=cfg.seed)
    selected_with_artefact_samples = non_ink_samples_with_artefact.sample(n=num_with_artefact_samples,
                                                                          random_state=cfg.seed)

    # Combine all selected samples
    return pd.concat([ink_samples, selected_no_artefact_samples,
                      selected_with_artefact_samples]), num_ink_samples, num_no_artefact_samples, num_with_artefact_samples


def generate_dataset(cfg: Config):
    csv_path = os.path.join(cfg.dataset_target_dir, str(cfg.patch_size), 'label_infos.csv')

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(e)
        sys.exit(1)

    if cfg.seed == -1:
        cfg.seed = None  # Set random seed if -1 is given

    count_zero = (df['ink_p'] == 0).sum()
    count_greater_than_zero = (df['ink_p'] > 0).sum()
    print(f"Before balancing: {count_zero} samples with ink_p = 0 and {count_greater_than_zero} samples with ink_p > 0")

    # BALANCING IS DONE ON CREATION
    # Step 1: Filter out rows where ink_p > ratio
    df_ink_p_greater_than_ink_ratio = df[df['ink_p'] > cfg.ink_ratio]

    # Step 2: Decide how many no-ink samples
    no_ink_sample_count = int(len(df_ink_p_greater_than_ink_ratio) * 0.2)

    # Step 3: Filter out rows where ink_p <= 0 and limit the number of rows
    max_ignore_percentage_th = 10
    df_good_no_inks = df[(df['ink_p'] == 0) & (df['ignore_p'] < max_ignore_percentage_th)].head(no_ink_sample_count)

    # # Step 4: Concatenate the two DataFrames
    df = pd.concat([df_ink_p_greater_than_ink_ratio, df_good_no_inks])

    count_zero = (df['ink_p'] == 0).sum()
    count_greater_than_zero = (df['ink_p'] > 0).sum()
    print(f"After balancing: {count_zero} samples with ink_p = 0 and {count_greater_than_zero} samples with ink_p > 0")
    # # Print statistics
    # print(f"Total ink samples: {num_ink_samples}")
    # print(f"Total non-ink samples with no artefact: {num_no_artefact_samples}")
    # print(f"Total non-ink samples with artefact > {cfg.artefact_threshold}: {num_with_artefact_samples}")
    # print(f"Total samples: {num_ink_samples + num_no_artefact_samples + num_with_artefact_samples}")

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
