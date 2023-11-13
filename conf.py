import albumentations as A
from albumentations.pytorch import ToTensorV2


class CFG:
    local = False

    data_root_dir = "data"
    fragment_root_dir = data_root_dir
    if local:
        fragment_root_dir = r"C:\Users\Marce\Git-Master\Privat\vesuv\data"

    data_out_path = "data/train"

    # ============== model cfg =============
    model_name = '3d_unet_segformer'
    backbone = 'None'
    # backbone = 'se_resnext50_32x4d'
    in_chans = 16

    # ============== training cfg =============
    size = 512
    tile_size = 512
    stride = tile_size // 4

    device = 'cuda'
    seed = 15

    epochs = 100
    lr = 1e-2
    train_batch_size = 32 # 32
    val_batch_size = 4
    num_workers = 1

    dataset_fraction = 0.25
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
        ], p=1.0),
        A.Normalize(mean=[0] * in_chans, std=[1] * in_chans),
        ToTensorV2(transpose_mask=True),
    ]

    valid_aug_list = [
        A.Resize(size, size),
        A.Normalize(mean=[0] * in_chans, std=[1] * in_chans),
        ToTensorV2(transpose_mask=True),
    ]



