import albumentations as A
weight_decay = 0.001
lr = 0.0001
eta_min = 1e-05
label_smoothing = 0.0
pos_weight = 1.0
optimizer = 'adamw'
JETFIRE_FRAG_ID = '20231005123336'
GRIMLARGE_FRAG_ID = '20231012184422'
THUNDERCRACKER_FRAG_ID = '20231012173610'
JAZZILLA_FRAG_ID = '20231016151001'
HOT_ROD_FRAG_ID = '20230929220926'
BLASTER_FRAG_ID = '20230702185753'
IRONHIDE_FRAG_ID = '20230905134255'
_base_ = ['configs/schedules/adamw_cosine_lr.py']
work_dir = ''
base_label_dir = 'data/base_label_files'
data_root_dir = 'data'
dataset_target_dir = 'multilayer_approach/datasets'
model_type = 'b3'
segformer_from_pretrained = 'nvidia/mit-b3'
architecture = 'unetr-sf'
model_name = 'unetr-sf-nvidia/mit-b3'
in_chans = 12
seed = 3445774
epochs = -1
losses = []
dataset_fraction = 1
unetr_out_channels = 32
val_interval = 1
patch_size = 512
label_size = 128
stride = 256
ink_ratio = 3
fragment_ids = ['20230702185753', '20230905134255', '20231012173610', '20231005123336', '20231012184422', '20231016151001']
validation_fragments = ['20230929220926']
train_split = 0.8
step_lr_steps = 1
step_lr_factor = 0.98
num_workers = 1
train_batch_size = 1
val_batch_size = 1
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
