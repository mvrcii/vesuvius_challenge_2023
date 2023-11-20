from lightning_modules.abstract_module import AbstractVesuvLightningModule
from models.cnn3d_segformer import CNN3D_Segformer


class CNN3D_SegformerModule(AbstractVesuvLightningModule):
    def __init__(self, cfg):
        super().__init__(cfg=cfg)
        self.model = CNN3D_Segformer()

    def forward(self, x):
        output = self.model(x.unsqueeze(1).float())

        return output
