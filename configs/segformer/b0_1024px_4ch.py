import os

_base_ = [
    "configs/datasets/single/ultra_magnus_1024px.py",
    "configs/schedules/adamw_cosine_lr.py",
]

work_dir = os.path.join("/scratch", "medfm", "vesuv", "kaggle1stReimp")
base_label_dir = os.path.join("data", "base_label_files")
data_root_dir = "data"
dataset_target_dir = os.path.join("data", "datasets")

model_type = "b0"
architecture = 'segformer'
model_name = f"{architecture}-{model_type}"
from_pretrained = f"nvidia/{model_type}-finetuned-cityscapes-1024-1024"
in_chans = 4
seed = 6346346
epochs = -1

dataset_fraction = 1
num_workers = 4
train_batch_size = 16
val_batch_size = 16
