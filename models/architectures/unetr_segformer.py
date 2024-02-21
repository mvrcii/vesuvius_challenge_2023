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



class UNETR_Segformer(nn.Module):
    def __init__(self, cfg, dropout=.2):
        super().__init__()
        self.cfg = cfg
        self.dropout = nn.Dropout2d(dropout)

        self.encoder = UNETR(
            epsilon=getattr(self.cfg, 'epsilon', 1e-3),
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
    # model = UNETR(input_dim=1, output_dim=32, img_shape=(16, 256, 256))
    model = UNETR_Segformer(CFG)

    if torch.cuda.is_available():
        model.to('cuda')
    else:
        print("CUDA is not available. The model will remain on the CPU.")

    # Input
    # x = torch.from_numpy(x).float()
    # x = torch.zeros(1, 1, 12, 128, 128)
    x = torch.empty((1, 1, 12, 128, 128)).fill_(float('inf'))

    if torch.isnan(x).any():
        print("Warning: Data is nan:", x)

    # pad to have depth 16 instead of 12
    x = torch.cat([x, torch.zeros(1, 1, 4, 128, 128)], dim=2)
    print(x.shape)

    print(torch.unique(x))

    # # move x to cuda
    x = x.to(get_device(model))
    output = model(x)

    print(output.shape)
    print(torch.unique(output))
