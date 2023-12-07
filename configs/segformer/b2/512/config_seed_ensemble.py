import os
import albumentations as A
from constants import (ULTRA_MAGNUS_FRAG_ID, OPTIMUS_FRAG_ID, IRONHIDE_FRAG_ID, MEGATRON_FRAG_ID,
                       BUMBLEBEE_FRAG_ID, SOUNDWAVE_FRAG_ID, STARSCREAM_FRAG_ID, RATCHET_FRAG_ID)

_base_ = [
    "configs/schedules/adamw_cosine_lr.py",
]

work_dir = os.path.join("/scratch", "medfm", "vesuv", "kaggle1stReimp")
base_label_dir = os.path.join("data", "base_label_files")
data_root_dir = "data"
dataset_target_dir = os.path.join("data", "datasets")

# dataset creation parameters
patch_size = 512
label_size = patch_size // 4
stride = patch_size // 2
ink_ratio = 3
artefact_threshold = 5
fragment_ids = [ULTRA_MAGNUS_FRAG_ID, OPTIMUS_FRAG_ID, BUMBLEBEE_FRAG_ID, MEGATRON_FRAG_ID, STARSCREAM_FRAG_ID,
                SOUNDWAVE_FRAG_ID, IRONHIDE_FRAG_ID, RATCHET_FRAG_ID]
train_split = 0.8

# training parameters
model_type = "b2"
architecture = 'segformer'
model_name = f"{architecture}-{model_type}"
from_pretrained = f"nvidia/mit-{model_type}"
# from_checkpoint = "kind-donkey-583-segformer-b2-231204-001337"
in_chans = 4
seed = 1
epochs = -1
losses = [("bce", 1.0), ("dice", 1.0)]
dataset_fraction = 1

val_interval = 1
lr = 5e-4

warmup_epochs = 5
warmup_start = 0.1
warmup_end = 1.0

step_lr_steps = 2
step_lr_factor = 0.99
weight_decay = 0.01

num_workers = 16
train_batch_size = 24
val_batch_size = 24

# TRAIN AUG AND VAL AUG HAVE TO BE LAST PARAMETERS OF CONFIG IN THIS ORDER
train_aug = [
    A.HorizontalFlip(),
    A.VerticalFlip(),
    A.RandomRotate90(),
    A.Transpose(),
    A.OneOf([
        A.RandomGamma(gamma_limit=(56, 150), eps=None),
        A.AdvancedBlur(blur_limit=(3, 5), sigmaX_limit=(0.2, 1.0), sigmaY_limit=(0.2, 1.0),
                       rotate_limit=(-90, 90), beta_limit=(0.5, 8.0), noise_limit=(0.9, 1.1)),
        A.CoarseDropout(max_holes=3, max_height=56, max_width=56, min_holes=2, min_height=38,
                        min_width=38, fill_value=0, mask_fill_value=None),
        A.Downscale(scale_min=0.65, scale_max=0.99),
        A.GridDistortion(num_steps=15, distort_limit=(-0.19, 0.19), interpolation=0,
                         border_mode=0,
                         value=(0, 0, 0), mask_value=None, normalized=False),
        A.RandomResizedCrop(height=patch_size, width=patch_size, scale=(0.65, 1.0),
                            ratio=(0.85, 1.15),
                            interpolation=0)
    ], p=0.5),
    A.ChannelDropout(p=0.05, channel_drop_range=(1, 1), fill_value=0),
    A.Normalize(mean=[0], std=[1])
]
val_aug = [
    A.Normalize(mean=[0], std=[1])
]
