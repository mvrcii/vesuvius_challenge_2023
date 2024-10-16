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
artefact_ratio = 5
fragment_ids = ['20231106155351', '20231024093300', '20230702185752_superseded', '20230522181603', '20230827161847', '20230904135535', '20230905134255', '20230909121925']
train_split = 0.8
model_type = 'b2'
architecture = 'segformer'
model_name = 'segformer-b2'
from_pretrained = 'nvidia/mit-b2'
in_chans = 4
seed = 7777
epochs = -1
loss = 'dice+bce'
dataset_fraction = 1
num_workers = 32
train_batch_size = 24
val_batch_size = 24
save = True
