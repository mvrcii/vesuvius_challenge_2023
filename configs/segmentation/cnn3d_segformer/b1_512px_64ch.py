import os

_base_ = [
    "configs/datasets/single/optimus_bumblebee_512px_64ch.py",
    "configs/schedules/adamw_cosine_lr.py",
]

segmentation = True
work_dir = os.path.join("/scratch", "medfm", "vesuv", "kaggle1stReimp")
base_label_dir = os.path.join("data", "base_label_files")
data_root_dir = "data"
dataset_target_dir = os.path.join("data", "datasets")

model_type = "b1"
architecture = 'cnn3d_segformer'
model_name = f"{architecture}-{model_type}"
from_pretrained = f"nvidia/segformer-{model_type}-finetuned-ade-512-512"
in_chans = 64
seed = 177
epochs = -1

dataset_fraction = 1
num_workers = 4
train_batch_size = 2
val_batch_size = 2
