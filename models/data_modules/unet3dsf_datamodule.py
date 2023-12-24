from models.data_modules.abstract_datamodule import AbstractDataModule
from models.datasets.unet3dsf_dataset import UNET3D_SFDataset


class UNET3D_SFDataModule(AbstractDataModule):
    def __init__(self, cfg):
        super().__init__(cfg=cfg)

    def get_dataset_class(self):
        return UNET3D_SFDataset
