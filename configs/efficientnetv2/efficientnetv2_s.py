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
seed = 235235
epochs = 50

num_workers = 4
train_batch_size = 4
val_batch_size = 4

lr = 2e-3
eta_min = 1e-4
weight_decay = 0.05
