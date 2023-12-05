from data_modules.abstract.abstract_datamodule import AbstractDataModule
from data_modules.abstract.abstract_dataset import AbstractDataset


class EfficientNetV2DataModule(AbstractDataModule):
    def __init__(self, cfg):
        super().__init__(cfg=cfg)

    def get_dataset_class(self):
        return AbstractDataset
