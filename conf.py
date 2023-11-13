import albumentations as A
from albumentations.pytorch import ToTensorV2


class CFG:
    local = False
    marcel = False  # only relevant if local=True

    data_root_dir = "data"
    fragment_root_dir = "/scratch/medfm/vesuv/kaggle1stReimp/data"
    if local:
        if marcel:
            fragment_root_dir = r"A:\projects_a\Python\vesuv\data"
        else:
            fragment_root_dir = r"C:\Users\Marce\Git-Master\Privat\vesuv\data"

    data_out_path = "data/train"

    # ============== model cfg =============
    in_chans = 16

    # ============== training cfg =============
    size = 512
    tile_size = 512
    stride = tile_size // 4

    device = 'cuda'
    seed = 15

    epochs = 20
    lr = 1e-4
    # Segformer b3, bs8 => 16GB VRAM
    train_batch_size = 8  # 32
    save_every_n_epochs = 3
    val_batch_size = 1
    num_workers = 0

    dataset_fraction = 1.0
    show_predictions = True

    # ============== fixed =============
    # min_lr = 1e-6
    # weight_decay = 1e-6
    # max_grad_norm = 1000

    # ============== augmentation =============

    # Train augmentations suitable for images + labels
    train_aug_list = [
        A.Resize(size, size),
        A.OneOf([
            A.HorizontalFlip(),
            A.VerticalFlip(),
            A.RandomRotate90(),
            A.Transpose(),
        ], p=0.25),
        A.Normalize(mean=[0] * in_chans, std=[1] * in_chans),
        ToTensorV2(transpose_mask=True),
    ]

    valid_aug_list = [
        A.Resize(size, size),
        A.Normalize(mean=[0] * in_chans, std=[1] * in_chans),
        ToTensorV2(transpose_mask=True),
    ]



