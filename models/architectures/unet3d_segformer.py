import albumentations as A
import torch
import torch.nn as nn
from albumentations.pytorch import ToTensorV2
from transformers import SegformerForSemanticSegmentation, SegformerConfig
import numpy as np

from models.architectures.unetr_segformer import UNETR_Segformer
from pytorch3dunet.unet3d.model import get_model

from models.architectures.unetr import UNETR


class CFG:
    # ============== comp exp name =============
    comp_name = 'vesuvius'

    # comp_dir_path = './'
    comp_dir_path = '/kaggle/input/'
    comp_folder_name = 'vesuvius-challenge-ink-detection'
    # comp_dataset_path = f'{comp_dir_path}datasets/{comp_folder_name}/'
    comp_dataset_path = f'{comp_dir_path}{comp_folder_name}/'
    patch_size = 256

    exp_name = '3d_unet_subv2'

    # ============== pred target =============
    target_size = 1

    # ============== model cfg =============
    model_name = '3d_unet_segformer'
    backbone = 'None'
    #     backbone = 'se_resnext50_32x4d'

    in_chans = 16
    # ============== training cfg =============
    # size = 64
    size = 256
    tile_size = 1024
    stride = tile_size // 4

    batch_size = 3  # 32
    use_amp = True

    scheduler = 'GradualWarmupSchedulerV2'
    # scheduler = 'CosineAnnealingLR'
    epochs = 15

    warmup_factor = 10
    lr = 1e-4 / warmup_factor

    # ============== fold =============
    valid_id = 2

    objective_cv = 'binary'  # 'binary', 'multiclass', 'regression'
    metric_direction = 'maximize'  # maximize, 'minimize'
    # metrics = 'dice_coef'

    # ============== fixed =============
    pretrained = True
    inf_weight = 'best'  # 'best'

    min_lr = 1e-6
    weight_decay = 1e-6
    max_grad_norm = 1000

    print_freq = 50
    num_workers = 2

    seed = 42

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


class UNET3D_Segformer(nn.Module):
    def __init__(self, cfg, dropout=.2):
        super().__init__()
        self.cfg = cfg
        self.dropout = nn.Dropout2d(dropout)

        self.encoder = get_model({"name": "UNet3D", "in_channels": 1, "out_channels": self.cfg.unet3d_out_channels, "f_maps": 8, "num_groups": 4,
                                  "is_segmentation": False})

        self.encoder_2d = SegformerForSemanticSegmentation.from_pretrained(
            pretrained_model_name_or_path=self.cfg.segformer_from_pretrained,
            num_channels=self.cfg.unet3d_out_channels,
            ignore_mismatched_sizes=True,
            num_labels=1,
        )
        # self.upscaler1 = nn.ConvTranspose2d(
        #     1, 1, kernel_size=(4, 4), stride=2, padding=1)
        # self.upscaler2 = nn.ConvTranspose2d(
        #     1, 1, kernel_size=(4, 4), stride=2, padding=1)

        # self.batch_norm = nn.BatchNorm2d(16) <- 16 = number of feature maps/channels (applied channel-wise)
        # self.batch_norm_upscale1 = nn.BatchNorm2d(..)
        # self.batch_norm_upscale2 = nn.BatchNorm2d(..)

    def forward(self, image):
        output = self.encoder(image)
        max_output = output.max(axis=2)
        final = max_output[0]
        print("SHAPES")
        print(output.shape)
        print(max_output.shape)
        print(final.shape)
        exit()

        output = self.encoder(image).max(axis=2)[0]  # 512, 512, 16 -> 512, 512, 16 (16 channels/feature maps??)
        output = self.encoder(image).mean(axis=2)[0]  # 512, 512, 16 -> 512, 512, 16 (16 channels/feature maps??)
        # TODO: Add BatchNorm2d/3D
        # TODO: output = self.batch_norm(output)
        output = self.dropout(output)
        output = self.encoder_2d(output).logits
        # output = self.upscaler1(output)
        # TODO: Add BatchNorm2d/3D
        # TODO: output = self.batch_norm_upscale1(output)
        # output = self.upscaler2(output)
        # TODO: Add BatchNorm2d/3D
        # TODO: output = self.batch_norm_upscale2(output)

        output = output.squeeze(1)

        return output

    def on_after_backward(self):
        for name, param in self.named_parameters():
            if param.grad is not None and torch.isnan(param.grad).any():
                print(f"NaN gradient in {name}")


def get_device(model):
    return next(model.parameters()).device


if __name__ == "__main__":
    # model = UNETR(input_dim=1, output_dim=32, img_shape=(16, 256, 256))
    model = UNETR_Segformer(CFG)
    #
    if torch.cuda.is_available():
        model.to('cuda')
    else:
        print("CUDA is not available. The model will remain on the CPU.")
    #
    # print(get_device(model))
    #
    # x = np.random.rand(1, 1, 16, 512, 512)
    # x = torch.from_numpy(x).float()
    x = torch.randn(1, 1, 12, 256, 256)
    print(x.shape)
    # pad to have depth 16 instead of 12
    x = torch.cat([x, torch.zeros(1, 1, 4, 256, 256)], dim=2)
    print(x.shape)
    # # move x to cuda
    x = x.to(get_device(model))
    output = model(x)
    print(output.shape)
