import os

import albumentations as A

from utility.fragments import (IRONHIDE_FRAG_ID, BLASTER_FRAG_ID,
                               THUNDERCRACKER_FRAG_ID, JETFIRE_FRAG_ID, GRIMHUGE_FRAG_ID, JAZZBIGGER_FRAG_ID,
                               SKYBIGGER_FRAG_ID, DEVASBIGGER_FRAG_ID, HOT_ROD_FRAG_ID, SUNSTREAKER_FRAG_ID,
                               ULTRA_MAGNUS_FRAG_ID, BLUEBIGGER_FRAG_ID, SKYHUGE_FRAG_ID, TRAILBIGGER_FRAG_ID,
                               SKYGLORIOUS_FRAG_ID)

_base_ = [
    "configs/schedules/adamw_cosine_lr.py",
]

# PATHS
# base_label_dir = os.path.join("data", "base_label_files")
data_root_dir = "data"


# CONTRASTED DATASET
contrasted = True
dataset_target_dir = os.path.join("data", "datasets", "segformer-b2_contrasted")


# BASE DATA - FRAGMENTS (all included here)
fragment_ids = [BLASTER_FRAG_ID, IRONHIDE_FRAG_ID, HOT_ROD_FRAG_ID, JETFIRE_FRAG_ID,
                SKYHUGE_FRAG_ID, SKYGLORIOUS_FRAG_ID, THUNDERCRACKER_FRAG_ID, GRIMHUGE_FRAG_ID,
                JAZZBIGGER_FRAG_ID, DEVASBIGGER_FRAG_ID, SUNSTREAKER_FRAG_ID, ULTRA_MAGNUS_FRAG_ID,
                BLUEBIGGER_FRAG_ID, TRAILBIGGER_FRAG_ID]
validation_fragments = [BLASTER_FRAG_ID]


# DATASET CREATION
in_chans = 12
patch_size = 64
label_size = patch_size // 4
stride = patch_size


# DATASET
dataset_fraction = 1.0
take_full_dataset = False
# Only relevant if take_full_dataset is False:
ink_ratio = 5  # Minimum amount of ink percentage in a patch
no_ink_sample_percentage = 1


# MODEL
model_type = "b5"
architecture = 'segformer'
model_name = f"{architecture}-{model_type}"
segformer_from_pretrained = f"nvidia/mit-{model_type}"


# TRAINING
seed = 7340043
epochs = -1
val_interval = 1


# OPTIMIZER
lr = 1e-4
step_lr_steps = 1
step_lr_factor = 0.98
weight_decay = 0.001
epsilon = 1e-3


# LOSS
# losses = [("bce", 1.0), ("dice", 1.0)]
losses = [('masked-dice', 1.0), ('masked-focal', 1.0)]
focal_gamma = 2.0
focal_alpha = 0.75
dice_smoothing = 0.05


# WORKER
node = False  # Model Checkpoints are only saved on a node
num_workers = 16
train_batch_size = 16
val_batch_size = train_batch_size


# AUGMENTATION
# TRAIN AUG AND VAL AUG HAVE TO BE LAST PARAMETERS OF CONFIG IN THIS ORDER
z_augment = False
train_aug = [
    A.RandomResizedCrop(height=patch_size, width=patch_size, scale=(0.78, 1.0),
                        ratio=(0.75, 1.51), interpolation=0, p=0.25),
    A.HorizontalFlip(p=0.25),
    A.VerticalFlip(p=0.25),
    A.RandomRotate90(p=0.25),
    A.ChannelDropout(p=0.2, channel_drop_range=(1, 4), fill_value=0),
    # A.OneOf([
    #     A.HorizontalFlip(),
    #     A.VerticalFlip(),
    #     A.RandomRotate90(),
    #     A.Transpose(),
    #     A.RandomGamma(always_apply=True, gamma_limit=(56, 150), eps=None),
    #     A.AdvancedBlur(always_apply=True, blur_limit=(3, 5), sigmaX_limit=(0.2, 1.0), sigmaY_limit=(0.2, 1.0),
    #                    rotate_limit=(-90, 90), beta_limit=(0.5, 8.0), noise_limit=(0.9, 1.1)),
    #     A.ChannelDropout(always_apply=True, channel_drop_range=(1, 1), fill_value=0),
    #     A.CoarseDropout(always_apply=True, max_holes=6, max_height=56, max_width=56, min_holes=2, min_height=38,
    #                     min_width=38, fill_value=0, mask_fill_value=None),
    #     A.Downscale(always_apply=True, scale_min=0.55, scale_max=0.99, interpolation=0),
    #     A.GridDistortion(always_apply=True, num_steps=15, distort_limit=(-0.19, 0.19), interpolation=0,
    #                      border_mode=0,
    #                      value=(0, 0, 0), mask_value=None, normalized=False),
    #     A.ImageCompression(always_apply=True, quality_lower=62, quality_upper=91, compression_type=1),
    #     A.RandomResizedCrop(always_apply=True, height=patch_size, width=patch_size, scale=(0.78, 1.0),
    #                         ratio=(0.75, 1.51),
    #                         interpolation=0)
    # ], p=0.5),
    A.Normalize(mean=[0], std=[1])
]
val_aug = [
    A.Normalize(mean=[0], std=[1])
]
