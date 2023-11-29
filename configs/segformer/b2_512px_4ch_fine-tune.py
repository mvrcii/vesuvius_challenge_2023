import os

_base_ = [
    "configs/datasets/single/optimus_starscream_ultra_magnus_512px.py",
    "configs/schedules/adamw_cosine_lr.py",
]

work_dir = os.path.join("/scratch", "medfm", "vesuv", "kaggle1stReimp")
base_label_dir = os.path.join("data", "base_label_files")
data_root_dir = "data"
dataset_target_dir = os.path.join("data", "datasets")

model_type = "b2"
architecture = 'segformer'
model_name = f"{architecture}-{model_type}"
from_pretrained = f"nvidia/mit-{model_type}"
from_checkpoint = "fine-wildflower-497-segformer-b2-231128-164424"
in_chans = 4
seed = 52352235
epochs = -1

dataset_fraction = 1
num_workers = 16
train_batch_size = 24
val_batch_size = 24

lr = 1e-5
eta_min = 1e-6
