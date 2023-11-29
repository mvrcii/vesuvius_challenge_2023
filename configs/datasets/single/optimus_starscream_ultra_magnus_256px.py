from constants import ULTRA_MAGNUS_FRAG_ID, OPTIMUS_FRAG_ID, STARSCREAM_FRAG_ID
import albumentations as A

calc_mean_std = False

patch_size = 256
label_size = patch_size // 4
stride = patch_size // 2
ink_ratio = 0.05
min_negative_patches_per_channel_block = 20

k_fold = False
train_frag_ids = [OPTIMUS_FRAG_ID, STARSCREAM_FRAG_ID, ULTRA_MAGNUS_FRAG_ID]
val_frag_ids = []
train_split = 0.8

train_aug = [
    A.OneOf([
        A.HorizontalFlip(),
        A.VerticalFlip(),
        A.RandomRotate90(),
        A.Transpose(),
    ], p=0.5),
    A.RandomResizedCrop(height=patch_size, width=patch_size, scale=(0.4, 0.9), ratio=(1, 1), p=0.5),
    # A.RandomScale(scale_limit=0.1, p=0.5),
    # A.CenterCrop(height=size, width=size, p=0.5),
]

val_aug = [
]

