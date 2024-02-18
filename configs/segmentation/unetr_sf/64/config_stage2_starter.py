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
work_dir = os.path.join("/scratch", "medfm", "vesuv", "MT3")
base_label_dir = os.path.join("data", "base_label_files")
data_root_dir = "data"


# CONTRASTED DATASET
contrasted = True
dataset_target_dir = os.path.join("data", "datasets", "unetr")
#contrasted = True
#dataset_target_dir = os.path.join("multilayer_approach", "datasets", "unetr_contrasted")


# BASE DATA - FRAGMENTS (all included here)
# fragment_ids = [BLASTER_FRAG_ID, IRONHIDE_FRAG_ID, HOT_ROD_FRAG_ID, JETFIRE_FRAG_ID,
#                 SKYHUGE_FRAG_ID, SKYGLORIOUS_FRAG_ID, THUNDERCRACKER_FRAG_ID, GRIMHUGE_FRAG_ID,
#                 JAZZBIGGER_FRAG_ID, DEVASBIGGER_FRAG_ID, SUNSTREAKER_FRAG_ID, ULTRA_MAGNUS_FRAG_ID,
#                 BLUEBIGGER_FRAG_ID, TRAILBIGGER_FRAG_ID]
fragment_ids = [GRIMHUGE_FRAG_ID]
validation_fragments = [GRIMHUGE_FRAG_ID]


# DATASET CREATION
in_chans = 12
patch_size = 64
label_size = patch_size // 4
stride = patch_size // 2


# DATASET
dataset_fraction = 1.0
take_full_dataset = False
# Only relevant if take_full_dataset is False:
ink_ratio = 15
no_ink_sample_percentage = 1


# MODEL
model_type = "b5"
architecture = 'unetr-sf'
unetr_out_channels = 32
model_name = f"{architecture}-{model_type}"
segformer_from_pretrained = f"nvidia/mit-{model_type}"


# TRAINING
seed = 7340043
epochs = -1
val_interval = 1


# OPTIMIZER
lr = 2e-5  # baseline is 1e-4
step_lr_steps = 1
step_lr_factor = 0.98
weight_decay = 0.001
epsilon = 1e-3


# LOSS
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
    A.Normalize(mean=[0], std=[1])
]
val_aug = [
    A.Normalize(mean=[0], std=[1])
]
