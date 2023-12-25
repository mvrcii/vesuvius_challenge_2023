import os
import albumentations as A
from utility.fragments import JAZZILLA_FRAG_ID, JETFIRE_FRAG_ID

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
ink_ratio = 50  # everything above this will be classified as 1
no_ink_sample_percentage = 1
max_ignore_th = 50
take_full_dataset = False

# training parameters
model_type = "regular"
architecture = 'vit3d'
model_name = f"{architecture}-{model_type}"
seed = 97074
epochs = -1
dataset_fraction = 0.5
val_interval = 1
fragment_ids = [JAZZILLA_FRAG_ID, JETFIRE_FRAG_ID]
# fragment_ids = [BLASTER_FRAG_ID, IRONHIDE_FRAG_ID, THUNDERCRACKER_FRAG_ID, JETFIRE_FRAG_ID, GRIMLARGE_FRAG_ID,
#                 JAZZILLA_FRAG_ID]
# validation_fragments = [HOT_ROD_FRAG_ID]
train_split = 0.8
lr = 1e-4
eta_min = 1e-5
step_lr_steps = 1
step_lr_factor = 0.98
weight_decay = 0.001
losses = [('mse', 1.0)]

num_workers = 8
train_batch_size = 8
val_batch_size = 8

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
