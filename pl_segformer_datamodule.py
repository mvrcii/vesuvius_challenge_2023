import os

from lightning.pytorch import LightningDataModule
from conf import CFG
from dataset import build_dataloader


class SegFormerDataModule(LightningDataModule):
    def __init__(self):
        super().__init__()

    def train_dataloader(self):
        return build_dataloader(
            data_root_dir=os.path.join(CFG.data_root_dir, str(CFG.size)),
            dataset_type='train')

    def val_dataloader(self):
        return build_dataloader(
            data_root_dir=os.path.join(CFG.data_root_dir, str(CFG.size)),
            dataset_type='val')

    # Implement test_dataloader if needed
