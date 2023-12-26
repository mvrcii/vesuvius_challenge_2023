import albumentations as A
import numpy as np
import torch
import torch.nn as nn
from albumentations.pytorch import ToTensorV2
from torch import optim
from transformers import SegformerForSemanticSegmentation

from models.architectures.unetr import UNETR


class CFG:
    segformer_from_pretrained = "nvidia/mit-b5"
    patch_size = 128
    unetr_out_channels = 32

    # ============== augmentation =============
    # train_aug_list = [
    #     # A.RandomResizedCrop(
    #     #     size, size, scale=(0.85, 1.0)),
    #     A.Resize(size, size),
    #     A.HorizontalFlip(p=0.5),
    #     A.VerticalFlip(p=0.5),
    #     A.RandomBrightnessContrast(p=0.75),
    #     A.ShiftScaleRotate(p=0.75),
    #     A.OneOf([
    #         A.GaussNoise(var_limit=[10, 50]),
    #         A.GaussianBlur(),
    #         A.MotionBlur(),
    #     ], p=0.4),
    #     A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.5),
    #     A.CoarseDropout(max_holes=1, max_width=int(size * 0.3), max_height=int(size * 0.3),
    #                     mask_fill_value=0, p=0.5),
    #     # A.Cutout(max_h_size=int(size * 0.6),
    #     #          max_w_size=int(size * 0.6), num_holes=1, p=1.0),
    #     A.Normalize(
    #         mean=[0] * in_chans,
    #         std=[1] * in_chans
    #     ),
    #     ToTensorV2(transpose_mask=True),
    # ]
    #
    # valid_aug_list = [
    #     A.Resize(size, size),
    #     A.Normalize(
    #         mean=[0] * in_chans,
    #         std=[1] * in_chans
    #     ),
    #     ToTensorV2(transpose_mask=True),
    # ]


class UNETR_Segformer(nn.Module):
    def __init__(self, cfg, dropout=.2):
        super().__init__()
        self.cfg = cfg
        self.dropout = nn.Dropout2d(dropout)

        self.encoder = UNETR(
            input_dim=1,
            output_dim=self.cfg.unetr_out_channels,
            img_shape=(16, self.cfg.patch_size, self.cfg.patch_size)
        )

        self.encoder_2d = SegformerForSemanticSegmentation.from_pretrained(
            pretrained_model_name_or_path=self.cfg.segformer_from_pretrained,
            num_channels=self.cfg.unetr_out_channels,
            ignore_mismatched_sizes=True,
            num_labels=1,
        )

    def forward(self, image):
        output = self.encoder(image).max(axis=2)[0]

        if torch.isnan(output).any():
            print("Warning: UNETR output is nan:", output)

        output = self.dropout(output)

        if torch.isnan(output).any():
            print("Warning: Dropout output is nan:", output)

        output = self.encoder_2d(output).logits

        if torch.isnan(output).any():
            print("Warning: Segformer logits output is nan:", output)

        output = output.squeeze(1)

        return output


def get_device(model):
    return next(model.parameters()).device


if __name__ == "__main__":
    # Assuming UNETR_Segformer and CFG are defined elsewhere
    model = UNETR_Segformer(CFG)

    if torch.cuda.is_available():
        model.to('cuda')
    else:
        print("CUDA is not available. The model will remain on the CPU.")

    # Define an optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Artificial loss tensor with NaN values
    loss = torch.tensor(float('nan'), requires_grad=True).to(get_device(model))
    print("Artificial loss:", loss)

    # Perform a backward step on the artificial loss
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Create a proper input tensor
    proper_input = torch.randn(1, 1, 16, 128, 128).to(get_device(model))

    # Check if the model gives proper output
    with torch.no_grad():
        proper_output = model(proper_input)
        print("Output shape:", proper_output.shape)
        print("Output unique values:", torch.unique(proper_output))