import os
import albumentations as A
weight_decay = 0.001
lr = 2e-05
eta_min = 1e-05
label_smoothing = 0.0
pos_weight = 1.0
optimizer = 'adamw'
IRONHIDE_FRAG_ID = '20230905134255'
BLASTER_FRAG_ID = '20230702185753'
THUNDERCRACKER_FRAG_ID = '20231012173610'
JETFIRE_FRAG_ID = '20231005123336'
GRIMHUGE_FRAG_ID = '20231012184423'
JAZZBIGGER_FRAG_ID = '20231016151002'
DEVASBIGGER_FRAG_ID = '20231022170901'
HOT_ROD_FRAG_ID = '20230929220926'
SUNSTREAKER_FRAG_ID = '20231031143852'
ULTRA_MAGNUS_FRAG_ID = '20231106155351'
BLUEBIGGER_FRAG_ID = '20231210121321'
SKYHUGE_FRAG_ID = '20231007101617'
TRAILBIGGER_FRAG_ID = '20231221180251'
SKYGLORIOUS_FRAG_ID = '20231007101619'
_base_ = ['configs/schedules/adamw_cosine_lr.py']
work_dir = '/mnt/vesuv/home/nowak/kaggle1stReimp'
base_label_dir = 'data/base_label_files'
data_root_dir = 'data'
contrasted = True
dataset_target_dir = 'multilayer_approach/datasets/unetr_contrasted'
segmentation = True
in_chans = 12
patch_size = 128
label_size = 32
stride = 64
fragment_ids = ['20230702185753', '20230905134255', '20230929220926', '20231005123336', '20231007101617', '20231007101619', '20231012173610', '20231012184423', '20231016151002', '20231022170901', '20231031143852', '20231106155351', '20231210121321', '20231221180251']
validation_fragments = ['20231210121321']
z_augment = True
model_type = 'b5'
segformer_from_pretrained = 'nvidia/mit-b5'
architecture = 'unetr-sf'
model_name = 'unetr-sf-b5'
dataset_fraction = 1.0
take_full_dataset = False
ink_ratio = 15
no_ink_sample_percentage = 0.75
seed = 43
epochs = -1
unetr_out_channels = 48
val_interval = 1
step_lr_steps = 1
step_lr_factor = 0.98
epsilon = 0.001
losses = [('masked-dice', 1.0), ('masked-focal', 4)]
focal_gamma = 3.0
focal_alpha = 0.85
dice_smoothing = 0.05
num_workers = 16
train_batch_size = 16
val_batch_size = 16
node = True
save = True

# TRAIN AUG AND VAL AUG HAVE TO BE LAST PARAMETERS OF CONFIG IN THIS ORDER
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
