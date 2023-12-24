from models.data_modules.abstract_datamodule import AbstractDataModule
from models.datasets.segformer_dataset import SegFormerDataset


class SegFormerDataModule(AbstractDataModule):
    def __init__(self, cfg):
        super().__init__(cfg=cfg)

    def get_dataset_class(self):
        return SegFormerDataset
