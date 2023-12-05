import os

from torch.utils.data import DataLoader

from data_modules.abstract.abstract_datamodule import AbstractDataModule
from data_modules.segformer.segformer_dataset import SegFormerDataset


class SegFormerDataModule(AbstractDataModule):
    def __init__(self, cfg):
        super().__init__(cfg=cfg)
        self.label_size = cfg.label_size

    def get_dataset_class(self):
        return SegFormerDataset

    def build_dataloader(self, dataset_type):
        if dataset_type == 'train':
            images_list = self.t_img_paths
            label_list = self.t_label_paths
        else:
            images_list = self.v_img_paths
            label_list = self.v_label_paths

        transform = self.get_transforms(dataset_type=dataset_type)
        root_dir = os.path.join(self.cfg.dataset_target_dir, str(self.cfg.patch_size))
        dataset = self.dataset(root_dir=root_dir,
                               images=images_list,
                               labels=label_list,
                               label_size=self.label_size,
                               transform=transform)

        batch_size = self.cfg.train_batch_size if dataset_type == 'train' else self.cfg.val_batch_size
        shuffle = dataset_type == 'train'
        data_loader = DataLoader(dataset,
                                 batch_size=batch_size,
                                 shuffle=shuffle,
                                 num_workers=self.cfg.num_workers,
                                 pin_memory=True,
                                 drop_last=False,
                                 persistent_workers=True)

        return data_loader
