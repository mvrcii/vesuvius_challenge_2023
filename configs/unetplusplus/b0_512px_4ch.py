import os

_base_ = [
    "configs/datasets/single/ultra_magnus_512px.py",
    "configs/schedules/adamw_cosine_lr.py",
]

work_dir = os.path.join("/scratch", "medfm", "vesuv", "kaggle1stReimp")
base_label_dir = os.path.join("data", "base_label_files")
data_root_dir = "data"
dataset_target_dir = os.path.join("data", "datasets")

# Encoder List: https://smp.readthedocs.io/en/latest/encoders.html
# encoder_name_and_weights = ("resnet34", "imagenet")
encoder_name_and_weights = ("timm-resnest14d", "imagenet")

architecture = 'unetplusplus'
model_name = f"{architecture}"
in_chans = 4
seed = 187
epochs = 100

dataset_fraction = 1
num_workers = 16
train_batch_size = 32
val_batch_size = 32
