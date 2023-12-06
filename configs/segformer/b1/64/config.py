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
patch_size = 64
label_size = patch_size // 4
stride = patch_size
ink_ratio = 25
artefact_threshold = 25
fragment_ids = [ULTRA_MAGNUS_FRAG_ID, OPTIMUS_FRAG_ID, BUMBLEBEE_FRAG_ID, MEGATRON_FRAG_ID, STARSCREAM_FRAG_ID,
                SOUNDWAVE_FRAG_ID, IRONHIDE_FRAG_ID, RATCHET_FRAG_ID]
train_split = 0.7

# training parameters
model_type = "b1"
architecture = 'segformer'
model_name = f"{architecture}-{model_type}"
from_pretrained = f"nvidia/mit-{model_type}"
in_chans = 16
seed = 21323
epochs = -1
losses = [("bce", 1.0), ("dice", 1.0)]
dataset_fraction = 1

val_interval = 5
lr = 1e-4
step_lr_steps = 2
step_lr_factor = 0.99
weight_decay = 0.005

num_workers = 16
train_batch_size = 16
val_batch_size = 16

train_aug = [
    A.OneOf([
        A.HorizontalFlip(),
        A.VerticalFlip(),
        A.RandomRotate90(),
        A.Transpose(),
        A.RandomGamma(always_apply=True, gamma_limit=(56, 150), eps=None),
        A.AdvancedBlur(always_apply=True, blur_limit=(3, 5), sigmaX_limit=(0.2, 1.0), sigmaY_limit=(0.2, 1.0),
                       rotate_limit=(-90, 90), beta_limit=(0.5, 8.0), noise_limit=(0.9, 1.1)),
        A.ChannelDropout(always_apply=True, channel_drop_range=(2, 4), fill_value=0),
    ], p=0.5),
    A.Normalize(mean=[0], std=[1])
]
val_aug = [
    A.Normalize(mean=[0], std=[1])
]
