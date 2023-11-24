import os

_base_ = [
    "configs/datasets/single/ultra_magnus_512px.py",
    "configs/schedules/adamw_cosine_lr.py",
]

work_dir = os.path.join("/scratch", "medfm", "vesuv", "kaggle1stReimp")
base_label_dir = os.path.join("data", "base_label_files")
data_root_dir = "data"
dataset_target_dir = os.path.join("data", "datasets")

model_type = "base"
architecture = 'unetplusplus'
model_name = f"{architecture}-{model_type}"
from_pretrained = "imagenet"
in_chans = 4
seed = 187
epochs = 150

dataset_fraction = 1
num_workers = 8
train_batch_size = 16
val_batch_size = 16
