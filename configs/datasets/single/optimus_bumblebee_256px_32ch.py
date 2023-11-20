from constants import OPTIMUS_FRAG_ID, BUMBLEBEE_FRAG_ID, ULTRA_MAGNUS_FRAG_ID
import albumentations as A

calc_mean_std = False

patch_size = 256
stride = patch_size // 2
dataset_in_chans = 32
ink_ratio = 0.01

k_fold = False
train_frag_ids = [OPTIMUS_FRAG_ID, BUMBLEBEE_FRAG_ID, ULTRA_MAGNUS_FRAG_ID]
val_frag_ids = []
train_split = 0.8

use_cutmix = False
use_mixup = False

# Train augmentations suitable for images + labels
train_common_aug = [
    A.OneOf([
        A.HorizontalFlip(),
        A.VerticalFlip(),
        A.RandomRotate90(),
        A.Transpose(),
    ], p=0.5),
    A.Normalize(mean=[0] * dataset_in_chans, std=[1] * dataset_in_chans),
]

train_image_aug = [
    # Scale = Percentage of images (min, max); Ratio (1, 1) = Square Crop
    A.RandomResizedCrop(height=patch_size, width=patch_size, scale=(0.4, 0.9), ratio=(1, 1), p=0.5),
    # A.OneOf([
    #     A.OpticalDistortion(p=0.5),
    #     A.GridDistortion(p=0.5),
    # ], p=0.25),
    # A.RandomScale(scale_limit=0.1, p=0.5),
    # A.CenterCrop(height=size, width=size, p=0.5),
    # A.Normalize(mean=[0] * in_chans, std=[1] * in_chans),
]

val_common_aug = [
]

val_image_aug = [
    A.Normalize(mean=[0] * dataset_in_chans, std=[1] * dataset_in_chans),
]
