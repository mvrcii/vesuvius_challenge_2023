import os

import albumentations as A


class CFG:
    data_root_dir = "/scratch/medfm/vesuv/kaggle1stReimp/data"
    base_label_dir = "data/base_label_files"
    dataset_target_dir = os.path.join(data_root_dir, "datasets")

    # ============== model =============
    in_chans = 16
    seg_pretrained = "nvidia/mit-b5"
    SEGFORMER_OUTPUT_DIM = (128, 128)
    """
    V-Ram Usage:
        Segformer b3:
            bs8 => 16GB (~3 min per epoch on gpu1b)
        Segformer B5:
            bs4 => 
    """

    # ============== training =============
    device = 'cuda'
    seed = 17
    epochs = 100

    # ========= optimizer =========
    weight_decay = 0.01
    lr = 1e-4

    # ============== dataset / dataloader =============
    calc_mean_std = False
    dataset_in_chans = 24
    patch_size = 512
    stride = patch_size // 2
    REQUIRED_LABEL_INK_PERCENTAGE = 0.1

    # single fragment dataset creation
    single_train_frag_id = 2
    train_split = 0.8

    # k_fold fragment dataset creation
    k_fold = True
    train_frag_ids = [2, 3, 4]
    val_frag_ids = [1]

    # ============ dataloader =============
    dataset_fraction = 1
    num_workers = 2
    train_batch_size = 2  # 32
    val_batch_size = 4

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
        ], p=0.5),
        # A.Normalize(mean=[0] * in_chans, std=[1] * in_chans),
    ]

    train_image_aug = [
        # TODO: CRASHING -> Resizing doesn't work with those augmentations somehow
        # A.RandomResizedCrop(height=size, width=size, p=0.5),
        # A.OneOf([
        #     A.OpticalDistortion(p=0.5),
        #     A.GridDistortion(p=0.5),
        # ], p=0.25),
        # A.RandomScale(scale_limit=0.1, p=0.5),
        # A.CenterCrop(height=size, width=size, p=0.5),
        # A.Normalize(mean=[0] * in_chans, std=[1] * in_chans),
    ]

    val_common_aug = [
    ]

    val_image_aug = [
        # A.Normalize(mean=[0] * in_chans, std=[1] * in_chans),
    ]
