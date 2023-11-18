import os

import albumentations as A


class CFG:
    data_root_dir = "/scratch/medfm/vesuv/kaggle1stReimp/data"
    base_label_dir = "data/base_label_files"
    dataset_target_dir = os.path.join(data_root_dir, "datasets")

    # ============== model =============
    in_chans = 64
    seg_pretrained = "nvidia/mit-b3"
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
    seed = 42
    epochs = 25

    # ========= optimizer =========
    weight_decay = 0.05
    lr = 1e-3

    # ============== dataset / dataloader =============
    calc_mean_std = False
    dataset_in_chans = 64
    patch_size = 512  # 64
    stride = patch_size // 4  # 64 / 2 = 32
    REQUIRED_LABEL_INK_PERCENTAGE = 0.01

    # single fragment dataset creation
    single_train_frag_id = 2
    train_split = 0.8

    # k_fold fragment dataset creation
    k_fold = True

    micha_frag = "20231024093300"
    marcel_frag = "20230702185752"

    # train_frag_ids = [
    #     "20230702185752",
    #     "20230827161847",
    #     "20230904135535",
    #     "20230905134255",
    #     "20230909121925"
    # ]
    # val_frag_ids = ["20230522181603"]

    train_frag_ids = [
        "20230522181603",
        "20230827161847",
        "20230904135535",
        "20230905134255",
        "20230909121925"]
    val_frag_ids = ["20230702185752"]

    # ============ dataloader =============
    dataset_fraction = 1
    num_workers = 10
    train_batch_size = 8  # 32
    val_batch_size = 8

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
        A.Normalize(mean=[0] * in_chans, std=[1] * in_chans),
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
        A.Normalize(mean=[0] * in_chans, std=[1] * in_chans),
    ]
