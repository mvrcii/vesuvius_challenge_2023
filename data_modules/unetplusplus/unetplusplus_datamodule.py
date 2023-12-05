from data_modules.abstract.abstract_datamodule import AbstractDataModule
from data_modules.unetplusplus.unetplusplus_dataset import UnetPlusPlusDataset


class UnetPlusPlusDataModule(AbstractDataModule):
    def __init__(self, cfg):
        super().__init__(cfg=cfg)

    def get_dataset_class(self):
        return UnetPlusPlusDataset
