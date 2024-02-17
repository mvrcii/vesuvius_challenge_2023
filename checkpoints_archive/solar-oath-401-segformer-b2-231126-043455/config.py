ULTRA_MAGNUS_FRAG_ID = '20231106155351'
OPTIMUS_FRAG_ID = '20231024093300'
STARSCREAM_FRAG_ID = '20230827161847'
calc_mean_std = False
patch_size = 512
label_size = 128
stride = 256
ink_ratio = 0.05
k_fold = False
train_frag_ids = ['20231024093300', '20230827161847', '20231106155351']
val_frag_ids = []
train_split = 0.8
val_common_aug = []
weight_decay = 0.01
lr = 0.0002
eta_min = 1e-05
label_smoothing = 0.0
pos_weight = 1.0
optimizer = 'adamw'
_base_ = ['configs/datasets/single/ultra_magnus_512px.py', 'configs/schedules/adamw_cosine_lr.py']
work_dir = '/scratch/medfm/vesuv/MT3'
base_label_dir = 'data/base_label_files'
data_root_dir = '/scratch/medfm/vesuv/MT3/data'
dataset_target_dir = '/scratch/medfm/vesuv/MT3/data/datasets'
model_type = 'b2'
architecture = 'segformer'
model_name = 'segformer-b2'
from_pretrained = 'nvidia/mit-b2'
in_chans = 4
seed = 240783
epochs = -1
dataset_fraction = 1
num_workers = 16
train_batch_size = 24
val_batch_size = 24
