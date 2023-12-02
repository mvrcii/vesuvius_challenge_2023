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
artefact_ratio = 5
# fragment_ids = [ULTRA_MAGNUS_FRAG_ID]
fragment_ids = [OPTIMUS_FRAG_ID, BUMBLEBEE_FRAG_ID, MEGATRON_FRAG_ID, STARSCREAM_FRAG_ID, SOUNDWAVE_FRAG_ID, IRONHIDE_FRAG_ID, RATCHET_FRAG_ID]
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

# training parameters
model_type = "b2"
architecture = 'segformer'
model_name = f"{architecture}-{model_type}"
from_pretrained = f"nvidia/mit-{model_type}"
# from_checkpoint = "fine-wildflower-497-segformer-b2-231128-164424"
in_chans = 4
seed = 52352235
epochs = -1
dataset_fraction = 0.1

num_workers = 4
train_batch_size = 4
val_batch_size = 4
