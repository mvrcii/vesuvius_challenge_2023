from models.abstract_model import AbstractVesuvLightningModule
from architectures.unetr_segformer import CFG, unet_3d_jumbo_config, UNETR_Segformer


class UNETR_SFModule(AbstractVesuvLightningModule):
    def __init__(self, cfg):
        super().__init__(cfg=cfg)
        self.model = UNETR_Segformer(CFG)
