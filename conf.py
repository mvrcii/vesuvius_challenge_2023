import albumentations as A
import torch.cuda
from albumentations.pytorch import ToTensorV2
from transformers import SegformerConfig


class CFG:
    local = True

    data_root_dir = "data"
    fragment_root_dir = r"C:\Users\Marce\Git-Master\Privat\vesuv\data"
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
    seed = 13

    epochs = 50
    lr = 1e-3
    train_batch_size = 1  # 32
    val_batch_size = 1
    num_workers = 1

    dataset_fraction = 0.1

    # ============== fixed =============
    # min_lr = 1e-6
    # weight_decay = 1e-6
    # max_grad_norm = 1000

    # ============== augmentation =============
    train_aug_list = [
        # A.RandomResizedCrop(
        #     size, size, scale=(0.85, 1.0)),
        A.Resize(size, size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.75),
        A.ShiftScaleRotate(p=0.75),
        A.OneOf([
            A.GaussNoise(var_limit=[10, 50]),
            A.GaussianBlur(),
            A.MotionBlur(),
        ], p=0.4),
        A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.5),
        A.CoarseDropout(max_holes=1, max_width=int(size * 0.3), max_height=int(size * 0.3),
                        mask_fill_value=0, p=0.5),
        # A.Cutout(max_h_size=int(size * 0.6),
        #          max_w_size=int(size * 0.6), num_holes=1, p=1.0),
        A.Normalize(
            mean=[0] * in_chans,
            std=[1] * in_chans
        ),
        ToTensorV2(transpose_mask=True),
    ]

    valid_aug_list = [
        A.Resize(size, size),
        A.Normalize(
            mean=[0] * in_chans,
            std=[1] * in_chans
        ),
        ToTensorV2(transpose_mask=True),
    ]


unet_3d_jumbo_config = SegformerConfig(
    **{
        "architectures": ["SegformerForImageClassification"],
        "attention_probs_dropout_prob": 0.0,
        "classifier_dropout_prob": 0.1,
        "decoder_hidden_size": 768,
        "depths": [3, 6, 40, 3],
        "downsampling_rates": [1, 4, 8, 16],
        "drop_path_rate": 0.1,
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.0,
        "hidden_sizes": [64, 128, 320, 512],
        "image_size": 224,
        "initializer_range": 0.02,
        "layer_norm_eps": 1e-06,
        "mlp_ratios": [4, 4, 4, 4],
        "model_type": "segformer",
        "num_attention_heads": [1, 2, 5, 8],
        "num_channels": 32,
        "num_encoder_blocks": 4,
        "patch_sizes": [7, 3, 3, 3],
        "sr_ratios": [8, 4, 2, 1],
        "strides": [4, 2, 2, 2],
        "torch_dtype": "float32",
        "transformers_version": "4.12.0.dev0",
        "num_labels": 1,
    }
)
