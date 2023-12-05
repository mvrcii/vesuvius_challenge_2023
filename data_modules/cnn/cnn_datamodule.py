from data_modules.abstract.abstract_datamodule import AbstractDataModule


class CNNDataModule(AbstractDataModule):
    def __init__(self, cfg):
        super().__init__(cfg=cfg)

    def get_dataset_class(self):
        return AbstractDataModule
