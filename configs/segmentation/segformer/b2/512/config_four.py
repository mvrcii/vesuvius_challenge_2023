import os
import albumentations as A
from utility.meta_data import AlphaBetaMeta

_base_ = [
    "configs/schedules/adamw_cosine_lr.py",
]

segmentation = True
# PATHS
work_dir = os.path.join("/scratch", "medfm", "vesuv", "kaggle1stReimp")
base_label_dir = os.path.join("data", "base_label_files")
data_root_dir = "data"
dataset_target_dir = os.path.join("data", "datasets")

# DATASET CREATION
in_chans = 4
patch_size = 512
label_size = patch_size // 4
stride = patch_size // 2
ink_ratio = 3
artefact_threshold = 5
fragment_ids = AlphaBetaMeta().get_current_train_fragments()
excluded_label_blocks = 3
excluded_label_layers = in_chans * excluded_label_blocks  # excluded from bottom and top of the stack
train_split = 0.8

# TRAINING
model_type = "b2"
architecture = 'segformer'
model_name = f"{architecture}-{model_type}"
from_pretrained = f"nvidia/mit-{model_type}"
# from_checkpoint = "kind-donkey-583-segformer-b2-231204-001337"
seed = 3445774
epochs = -1
losses = [("bce", 1.0), ("dice", 1.0)]
dataset_fraction = 1
val_interval = 2
lr = 1e-4
step_lr_steps = 1
step_lr_factor = 0.98
weight_decay = 0.001
num_workers = 16
train_batch_size = 24
val_batch_size = 24

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
        A.ImageCompression(always_apply=True, quality_lower=62, quality_upper=91, compression_type=1),
        A.RandomResizedCrop(always_apply=True, height=patch_size, width=patch_size, scale=(0.78, 1.0),
                            ratio=(0.75, 1.51),
                            interpolation=0)
    ], p=0.5),
    A.Normalize(mean=[0], std=[1])
]
val_aug = [
    A.Normalize(mean=[0], std=[1])
]
