from models.abstract_model import AbstractVesuvLightningModule
from models.architectures.unetr_segformer import UNETR_Segformer


class UNETR_SFModule(AbstractVesuvLightningModule):
    def __init__(self, cfg):
        super().__init__(cfg=cfg)
        self.model = UNETR_Segformer(cfg=cfg)

    def forward(self, x):
        output = self.model(x)
        return output
