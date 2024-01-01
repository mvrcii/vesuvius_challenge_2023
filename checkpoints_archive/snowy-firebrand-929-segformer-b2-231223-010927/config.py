import os
import sys
import albumentations as A
import cv2
weight_decay = 0.001
lr = 0.0001
eta_min = 1e-06
label_smoothing = 0.0
pos_weight = 1.0
optimizer = 'adamw'
_base_ = ['configs/schedules/adamw_cosine_lr.py']
data_root_dir = 'data'
dataset_target_dir = 'data/datasets'
model_type = 'b2'
architecture = 'segformer'
model_name = 'segformer-b2'
from_pretrained = 'nvidia/mit-b2'
in_chans = 1
seed = 9393
epochs = -1
losses = [('focal', 2.0), ('dice', 1.0)]
patch_size = 512
label_size = 128
stride = 256
ink_ratio = -1
no_ink_sample_percentage = 1
excluded_label_blocks = 0
excluded_label_layers = 0
fragment_ids = ['20231005123336', '20231012184422']
take_full_dataset = True
dataset_fraction = 1
train_split = 0.8
val_interval = 1
step_lr_steps = 1
step_lr_factor = 0.98
num_workers = 16
train_batch_size = 24
val_batch_size = 24
save = True
train_aug = [
    A.OneOf([
        A.HorizontalFlip(),
        A.VerticalFlip(),
        A.RandomRotate90(),
        A.Transpose(),
        A.RandomGamma(always_apply=True, gamma_limit=(56, 150), eps=None),
        A.AdvancedBlur(always_apply=True, blur_limit=(3, 5), sigmaX_limit=(0.2, 1.0), sigmaY_limit=(0.2, 1.0),
                       rotate_limit=(-90, 90), beta_limit=(0.5, 8.0), noise_limit=(0.9, 1.1)),
        A.CoarseDropout(always_apply=True, max_holes=6, max_height=56, max_width=56, min_holes=2, min_height=38,
                        min_width=38, fill_value=0, mask_fill_value=None),
        A.Downscale(always_apply=True, scale_min=0.55, scale_max=0.99, interpolation=cv2.INTER_LANCZOS4),
        A.GridDistortion(always_apply=True, num_steps=15, distort_limit=(-0.19, 0.19), interpolation=cv2.INTER_LANCZOS4,
                         border_mode=0,
                         value=(0, 0, 0), mask_value=None, normalized=False),
        A.RandomResizedCrop(always_apply=True, height=patch_size, width=patch_size, scale=(0.78, 1.0),
                            ratio=(0.75, 1.51),
                            interpolation=cv2.INTER_LANCZOS4)
    ], p=0.5),
    A.Normalize(mean=[0], std=[1])
]
val_aug = [
    A.Normalize(mean=[0], std=[1])
]
