import os
import sys

import albumentations as A

from utility.fragments import JETFIRE_FRAG_ID, GRIMLARGE_FRAG_ID, THUNDERCRACKER_FRAG_ID, JAZZILLA_FRAG_ID, HOT_ROD_FRAG_ID, \
    BLASTER_FRAG_ID, IRONHIDE_FRAG_ID

sys.path.append('../')

segmentation = True
_base_ = [
    "configs/schedules/adamw_cosine_lr.py",
]

work_dir = os.path.join("/scratch", "medfm", "vesuv", "kaggle1stReimp")
base_label_dir = os.path.join("data", "base_label_files")
data_root_dir = "data"
dataset_target_dir = os.path.join("multilayer_approach", "datasets")

# training parameters
model_type = "b3"
segformer_from_pretrained = f"nvidia/mit-{model_type}"
# from_pretrained = "playful-glade-812-unetr-sf-b3-231216-041654"
architecture = 'unetr-sf'
model_name = f"{architecture}-{model_type}"

in_chans = 12
seed = 770
epochs = -1
losses = []
dataset_fraction = 1
unetr_out_channels = 32

val_interval = 1

# dataset creation parameters
patch_size = 512
label_size = patch_size // 4
stride = patch_size // 2
ink_ratio = 8
fragment_ids = [BLASTER_FRAG_ID, IRONHIDE_FRAG_ID, THUNDERCRACKER_FRAG_ID, JETFIRE_FRAG_ID, GRIMLARGE_FRAG_ID,
                JAZZILLA_FRAG_ID]
validation_fragments = [HOT_ROD_FRAG_ID]
train_split = 0.8

lr = 1e-3  # 1e-4
step_lr_steps = 1
step_lr_factor = 0.98
weight_decay = 0.001
take_full_dataset = False
no_ink_sample_percentage = 0.2  # Given the count of N ink samples, take this percentage of the count as no ink
max_ignore_th = 75  # For no ink samples, how much ignore percentage (%) is allowed maximum

num_workers = 16
train_batch_size = 1
val_batch_size = train_batch_size

# TRAIN AUG AND VAL AUG HAVE TO BE LAST PARAMETERS OF CONFIG IN THIS ORDER
train_aug = [
    A.OneOf([
        A.HorizontalFlip(),
        A.VerticalFlip(),
        A.RandomRotate90(),
        A.Transpose(),
        A.RandomGamma(always_apply=True, gamma_limit=(56, 150), eps=None),
        A.AdvancedBlur(always_apply=True, blur_limit=(3, 5), sigmaX_limit=(0.2, 1.0), sigmaY_limit=(0.2, 1.0),
                       rotate_limit=(-90, 90), beta_limit=(0.5, 8.0), noise_limit=(0.9, 1.1)),
        A.ChannelDropout(always_apply=True, channel_drop_range=(1, 1), fill_value=0),
        A.CoarseDropout(always_apply=True, max_holes=6, max_height=56, max_width=56, min_holes=2, min_height=38,
                        min_width=38, fill_value=0, mask_fill_value=None),
        A.Downscale(always_apply=True, scale_min=0.55, scale_max=0.99),
        A.GridDistortion(always_apply=True, num_steps=15, distort_limit=(-0.19, 0.19), interpolation=0,
                         border_mode=0,
                         value=(0, 0, 0), mask_value=None, normalized=False),
        A.RandomResizedCrop(always_apply=True, height=patch_size, width=patch_size, scale=(0.78, 1.0),
                            ratio=(0.75, 1.51),
                            interpolation=0)
    ], p=0.1),
    # A.ChannelDropout(p=0.05, channel_drop_range=(1, 1), fill_value=0),
    A.Normalize(mean=[0], std=[1])
]
val_aug = [
    A.Normalize(mean=[0], std=[1])
]
