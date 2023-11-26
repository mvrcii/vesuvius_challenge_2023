_base_ = [
    "configs/datasets/slice/ultra_magnus_slice.py",
    "configs/schedules/adamw_cosine_lr.py",
]

work_dir = "/scratch/medfm/vesuv/kaggle1stReimp"
base_label_dir = "data/base_label_files"
data_root_dir = "data"
dataset_target_dir = "data/datasets"


model_type = "50"
architecture = 'resnet'
model_name = f"{architecture}-{model_type}"
in_chans = 1
seed = 123456
epochs = 50

num_workers = 4
train_batch_size = 64
val_batch_size = 64
