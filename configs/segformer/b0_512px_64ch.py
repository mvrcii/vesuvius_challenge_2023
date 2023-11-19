import os

_base_ = [
    "configs/datasets/single/optimus_bumblebee_512px_64ch.py",
    "configs/schedules/adamw_cosine_lr.py",
]

work_dir = os.path.join("/scratch", "medfm", "vesuv", "kaggle1stReimp")
base_label_dir = os.path.join("data", "base_label_files")
data_root_dir = "data"
dataset_target_dir = os.path.join("data", "datasets")

model_type = "b0"
architecture = 'segformer'
model_name = f"{architecture}-{model_type}"
from_pretrained = f"nvidia/mit-{model_type}"
in_chans = 64
seed = 3
epochs = 100

dataset_fraction = 1
num_workers = 4
train_batch_size = 4
val_batch_size = 4
