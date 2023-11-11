import torch
import torch.nn as nn
from transformers import SegformerForSemanticSegmentation

from conf import unet_3d_jumbo_config
from unetr import UNETR


class UNETR_Segformer(nn.Module):
    def __init__(self, cfg, dropout=.2):
        super().__init__()
        self.cfg = cfg
        self.dropout = nn.Dropout2d(dropout)
        self.encoder = UNETR(
            input_dim=1,
            output_dim=32,
            img_shape=(16, self.cfg.size, self.cfg.size)
        )
        self.encoder_2d = SegformerForSemanticSegmentation(unet_3d_jumbo_config)
        self.upscaler1 = nn.ConvTranspose2d(
            1, 1, kernel_size=(4, 4), stride=2, padding=1)
        self.upscaler2 = nn.ConvTranspose2d(
            1, 1, kernel_size=(4, 4), stride=2, padding=1)

        # self.batch_norm = nn.BatchNorm2d(16) <- 16 = number of feature maps/channels (applied channel-wise)
        # self.batch_norm_upscale1 = nn.BatchNorm2d(..)
        # self.batch_norm_upscale2 = nn.BatchNorm2d(..)

    def forward(self, image):
        output = self.encoder(image).max(axis=2)[0]  # 512, 512, 16 -> 512, 512, 16 (16 channels/feature maps??)
        # TODO: Add BatchNorm2d/3D
        # TODO: output = self.batch_norm(output)
        output = self.dropout(output)
        output = self.encoder_2d(output).logits
        output = self.upscaler1(output)
        # TODO: Add BatchNorm2d/3D
        # TODO: output = self.batch_norm_upscale1(output)
        output = self.upscaler2(output)
        # TODO: Add BatchNorm2d/3D
        # TODO: output = self.batch_norm_upscale2(output)

        output = output.squeeze(1)
        output = torch.sigmoid(output)

        return output
