import torch.nn as nn
from transformers import SegformerForSemanticSegmentation

from conf import CFG, unet_3d_jumbo_config
from unetr import UNETR
from dataset import make_test_dataset


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

    def forward(self, image):
        output = self.encoder(image).max(axis=2)[0]
        output = self.dropout(output)
        output = self.encoder_2d(output).logits
        output = self.upscaler1(output)
        output = self.upscaler2(output)
        return output


model = UNETR_Segformer(CFG)
test_loader, xyxys = make_test_dataset(2)