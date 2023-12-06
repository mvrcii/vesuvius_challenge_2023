import timm
from timm.data import resolve_data_config, create_transform

from data_modules.abstract.abstract_datamodule import AbstractDataModule
from data_modules.abstract.abstract_dataset import AbstractDataset


class InceptionDataModule(AbstractDataModule):
    def __init__(self, cfg):
        super().__init__(cfg=cfg)

        model = timm.create_model('inception_v4', pretrained=True)
        config = resolve_data_config({}, model=model)
        self.transform = create_transform(**config)

    def get_dataset_class(self):
        return AbstractDataset
