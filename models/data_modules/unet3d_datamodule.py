from models.data_modules.abstract_datamodule import AbstractDataModule
from models.datasets.unet3d_dataset import UNET3D_Dataset


class UNET3D_DataModule(AbstractDataModule):
    def __init__(self, cfg):
        super().__init__(cfg=cfg)

    def get_dataset_class(self):
        return UNET3D_Dataset
