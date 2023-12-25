import os
import albumentations as A
from utility.fragments import JAZZILLA_FRAG_ID, JETFIRE_FRAG_ID, IRONHIDE_FRAG_ID, BLASTER_FRAG_ID, \
    THUNDERCRACKER_FRAG_ID, GRIMLARGE_FRAG_ID

_base_ = [
    "configs/schedules/adamw_cosine_lr.py",
]

# PATHS
work_dir = os.path.join("/scratch", "medfm", "vesuv", "kaggle1stReimp")
base_label_dir = os.path.join("data", "base_label_files")
data_root_dir = "data"
dataset_target_dir = os.path.join("multilayer_approach", "datasets")

# DATASET CREATION
in_chans = 12
patch_size = 48
stride = patch_size // 2

# if used with create_dataset_classification all samples with ink_ratio == 0 and > ink_ratio will be saved
# E.g. ink_ratio == 50 => discard samples with ink_ratio 1-49. Label for a sample will be its ink ratio
ink_ratio = 25


no_ink_sample_percentage = 1
# if label_infos.csv contains 'ignore_p', all samples with > max_ignore_th will be removed
# (however the current create_dataset_classification currently already removes a sample if it has ignore_p > 0)
max_ignore_th = 50


take_full_dataset = False

# training parameters
model_type = "regular"
architecture = 'vit3d'
model_name = f"{architecture}-{model_type}"
seed = 1337
epochs = -1
dataset_fraction = 0.01
val_interval = 1
fragment_ids = [JAZZILLA_FRAG_ID, JETFIRE_FRAG_ID, IRONHIDE_FRAG_ID, BLASTER_FRAG_ID, THUNDERCRACKER_FRAG_ID]
# fragment_ids = [GRIMLARGE_FRAG_ID]
validation_fragments = [GRIMLARGE_FRAG_ID]
train_split = 0.8
lr = 5e-5
eta_min = 1e-5
step_lr_steps = 1
step_lr_factor = 0.95
weight_decay = 0.001
losses = [('mse', 1.0)]

num_workers = 4
train_batch_size = 2
val_batch_size = 2

# TRAIN AUG AND VAL AUG HAVE TO BE LAST PARAMETERS OF CONFIG IN THIS ORDER
train_aug = [
    # A.OneOf([
    #     A.HorizontalFlip(),
    #     A.VerticalFlip(),
    #     A.RandomRotate90(),
    #     A.Transpose(),
    # ], p=0.1),
    # A.ChannelDropout(p=0.05, channel_drop_range=(1, 1), fill_value=0),
    A.Normalize(mean=[0], std=[1])
]
val_aug = [
    A.Normalize(mean=[0], std=[1])
]
