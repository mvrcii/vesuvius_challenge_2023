import os
import albumentations as A
weight_decay = 0.01
lr = 0.0002
eta_min = 1e-05
label_smoothing = 0.0
pos_weight = 1.0
optimizer = 'adamw'
ULTRA_MAGNUS_FRAG_ID = '20231106155351'
OPTIMUS_FRAG_ID = '20231024093300'
IRONHIDE_FRAG_ID = '20230905134255'
MEGATRON_FRAG_ID = '20230522181603'
BUMBLEBEE_FRAG_ID = '20230702185752_superseded'
SOUNDWAVE_FRAG_ID = '20230904135535'
STARSCREAM_FRAG_ID = '20230827161847'
RATCHET_FRAG_ID = '20230909121925'
_base_ = ['configs/schedules/adamw_cosine_lr.py']
work_dir = ''
base_label_dir = 'data/base_label_files'
data_root_dir = 'data'
dataset_target_dir = 'data/datasets'
patch_size = 512
label_size = 128
stride = 256
ink_ratio = 3
artefact_threshold = 5
fragment_ids = ['20231106155351', '20231024093300', '20230702185752_superseded', '20230522181603', '20230827161847', '20230904135535', '20230905134255', '20230909121925']
train_split = 0.8
model_type = 'b2'
architecture = 'segformer'
model_name = 'segformer-b2'
from_pretrained = 'nvidia/mit-b2'
in_chans = 4
seed = 7777
epochs = -1
losses = [('bce', 1.0), ('dice', 1.0)]
dataset_fraction = 1
val_interval = 1
step_lr_steps = 2
step_lr_factor = 0.99
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
    # A.ChannelDropout(p=0.05, channel_drop_range=(1, 1), fill_value=0),
    A.Normalize(mean=[0], std=[1])
]
val_aug = [
    A.Normalize(mean=[0], std=[1])
]
