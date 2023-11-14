import albumentations as A
from albumentations.pytorch import ToTensorV2


class CFG:
    local = False
    marcel = False  # only relevant if local=True
    marcel_mac = False

    data_root_dir = "data"
    fragment_root_dir = "/scratch/medfm/vesuv/kaggle1stReimp/data"
    if local:
        if marcel:
            if marcel_mac:
                fragment_root_dir = r"/Users/marcel/Documents/Git-Master/Private/kaggle1stReimp/data"
            else:
                fragment_root_dir = r"C:\Users\Marce\Git-Master\Privat\vesuv\data"
        else:
            fragment_root_dir = r"A:\projects_a\Python\vesuv\data"

    data_out_path = "data/train"

    # ============== model =============
    in_chans = 16
    seg_pretrained = "nvidia/mit-b3"
    """
    V-Ram Usage:
        Segformer b3:
            bs8 => 16GB (~3 min per epoch on gpu1b)
    """

    # ============== training =============
    device = 'cuda'
    seed = 15
    epochs = 20

    # ========= optimizer =========
    weight_decay = 0.01
    lr = 1e-5

    # ============== dataset / dataloader =============
    size = 512
    tile_size = 512
    stride = tile_size // 4

    train_batch_size = 8  # 32
    val_batch_size = 4

    num_workers = 2
    dataset_fraction = 1

    save_every_n_epochs = 3
    show_predictions = True

    # ============== fixed =============
    # min_lr = 1e-6
    # weight_decay = 1e-6
    # max_grad_norm = 1000

    # ============== augmentation =============
    use_cutmix = False
    use_mixup = False
    use_aug = True

    # Train augmentations suitable for images + labels
    train_common_aug = [
        A.OneOf([
            A.HorizontalFlip(),
            A.VerticalFlip(),
            A.RandomRotate90(),
            A.Transpose(),
        ], p=0.25),
        # A.Normalize(mean=[0] * in_chans, std=[1] * in_chans),
    ]

    train_image_aug = [
        # A.Normalize(mean=[0] * in_chans, std=[1] * in_chans),
    ]

    val_common_aug = [
    ]

    val_image_aug = [
        # A.Normalize(mean=[0] * in_chans, std=[1] * in_chans),
    ]
