import timm
from lightning import LightningModule

from models.abstract_model import AbstractVesuvLightningModule


class InceptionModule(AbstractVesuvLightningModule):
    def __init__(self, cfg):
        super().__init__(cfg=cfg)
        self.lr = cfg.lr
        self.weight_decay = cfg.weight_decay
        self.epochs = cfg.epochs
        self.eta_min = cfg.eta_min

        self.model = timm.create_model(cfg.from_pretrained,
                                       in_chans=cfg.in_chans,
                                       pretrained=True,
                                       num_classes=1)
