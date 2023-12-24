from models.data_modules.abstract_datamodule import AbstractDataModule
from models.datasets.unetrsf_dataset import UNETR_SFDataset


class UNETR_SFDataModule(AbstractDataModule):
    def __init__(self, cfg):
        super().__init__(cfg=cfg)

    def get_dataset_class(self):
        return UNETR_SFDataset
