import os

import albumentations as A

from utility.fragments import (IRONHIDE_FRAG_ID, BLASTER_FRAG_ID,
                               THUNDERCRACKER_FRAG_ID, JETFIRE_FRAG_ID, GRIMHUGE_FRAG_ID, JAZZBIGGER_FRAG_ID,
                               SKYBIGGER_FRAG_ID, DEVASBIGGER_FRAG_ID, HOT_ROD_FRAG_ID, SUNSTREAKER_FRAG_ID,
                               ULTRA_MAGNUS_FRAG_ID, BLUEBIGGER_FRAG_ID)

_base_ = [
    "configs/schedules/adamw_cosine_lr.py",
]

# PATHS
work_dir = os.path.join("/scratch", "medfm", "vesuv", "kaggle1stReimp")
base_label_dir = os.path.join("data", "base_label_files")
data_root_dir = "data"
contrasted = True
dataset_target_dir = os.path.join("multilayer_approach", "datasets", "unetr_contrasted")

# MODEL TYPE
segmentation = True

# DATASET CREATION
in_chans = 12
patch_size = 512
label_size = patch_size // 4
stride = patch_size // 2

fragment_ids = [IRONHIDE_FRAG_ID, BLASTER_FRAG_ID, THUNDERCRACKER_FRAG_ID, JETFIRE_FRAG_ID,
                GRIMHUGE_FRAG_ID, JAZZBIGGER_FRAG_ID, DEVASBIGGER_FRAG_ID,
                HOT_ROD_FRAG_ID, SUNSTREAKER_FRAG_ID, ULTRA_MAGNUS_FRAG_ID, BLUEBIGGER_FRAG_ID]
validation_fragments = [SKYBIGGER_FRAG_ID]

# Training parameters
model_type = "b5"
segformer_from_pretrained = f"nvidia/mit-{model_type}"
from_checkpoint = "electric-eon-1151-unetr-sf-b5-231228-144234"
architecture = 'unetr-sf'
model_name = f"{architecture}-{model_type}"

dataset_fraction = 1.0
take_full_dataset = False
# Only relevant if take_full_dataset == False
ink_ratio = 5
no_ink_sample_percentage = 1

seed = 7340043
epochs = -1
unetr_out_channels = 32

val_interval = 1

lr = 1e-4  # 1e-4
step_lr_steps = 1
step_lr_factor = 0.98
weight_decay = 0.001
epsilon = 1e-3

losses = [('masked-dice', 1.0)]
focal_gamma = 2.0
focal_alpha = 0.75
dice_smoothing = 0.05

num_workers = 16
train_batch_size = 6
val_batch_size = train_batch_size

# TRAIN AUG AND VAL AUG HAVE TO BE LAST PARAMETERS OF CONFIG IN THIS ORDER
train_aug = [
    A.RandomResizedCrop(height=patch_size, width=patch_size, scale=(0.78, 1.0),
                        ratio=(0.75, 1.51), interpolation=0, p=0.25),
    A.HorizontalFlip(p=0.25),
    A.VerticalFlip(p=0.25),
    A.RandomRotate90(p=0.25),
    # A.RandomBrightnessContrast(p=0.25),
    # A.OneOf([
    #     A.GaussNoise(var_limit=[10, 50]),
    #     A.GaussianBlur(),
    #     A.MotionBlur(),
    # ], p=0.25),
    # A.RandomGamma(always_apply=True, gamma_limit=(56, 150), eps=None),
    # A.AdvancedBlur(always_apply=True, blur_limit=(3, 5), sigmaX_limit=(0.2, 1.0), sigmaY_limit=(0.2, 1.0),
    #                rotate_limit=(-90, 90), beta_limit=(0.5, 8.0), noise_limit=(0.9, 1.1)),
    # A.ChannelDropout(always_apply=True, channel_drop_range=(1, 1), fill_value=0),
    # A.Downscale(always_apply=True, scale_min=0.55, scale_max=0.99),
    # A.GridDistortion(p=0.05, num_steps=15, distort_limit=(-0.19, 0.19), interpolation=0,
    #                  border_mode=0, value=(0, 0, 0), mask_value=None, normalized=False),
    A.ChannelDropout(p=0.2, channel_drop_range=(1, 4), fill_value=0),
    A.Normalize(mean=[0], std=[1])
]
val_aug = [
    A.Normalize(mean=[0], std=[1])
]
