_base_ = [
    "configs/datasets/slice/ultra_magnus_slice.py",
    "configs/schedules/adamw_cosine_lr.py",
]

work_dir = "/scratch/medfm/vesuv/kaggle1stReimp"
base_label_dir = "data/base_label_files"
data_root_dir = "data"
dataset_target_dir = "data/datasets"


model_type = "s"
architecture = 'tf_efficientnetv2_s'
model_name = f"{architecture}-{model_type}"
from_pretrained = f"tf_efficientnetv2_s"
in_chans = 1
seed = 42
epochs = 100

num_workers = 4
train_batch_size = 2560
val_batch_size = 2560

lr = 1e-6
eta_min = 1e-7
weight_decay = 0.1
