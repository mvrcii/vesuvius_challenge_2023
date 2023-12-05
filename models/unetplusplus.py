import sys

import segmentation_models_pytorch as smp
from einops import rearrange

from models.abstract_model import AbstractVesuvLightningModule


class UnetPlusPlusModule(AbstractVesuvLightningModule):
    def __init__(self, cfg):
        super().__init__(cfg=cfg)

        encoder_name, encoder_weights = getattr(cfg, 'encoder_name_and_weights', None)

        if not encoder_weights or not encoder_name:
            print("Missing encoder_name_and_weights param in config")
            sys.exit(1)

        self.model = smp.UnetPlusPlus(
            encoder_name=encoder_name,  # Encoders with weights https://smp.readthedocs.io/en/latest/encoders.html
            encoder_weights=encoder_weights,
            in_channels=cfg.in_chans,
            classes=1,
        )

    def forward(self, x):
        output = self.model(x.float())
        output = rearrange(output, 'b 1 h w -> b h w')

        return output
