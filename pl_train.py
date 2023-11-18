import importlib.util
import os
import pprint
from datetime import datetime
import sys
import torch
from lightning import seed_everything
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.trainer import Trainer

from conf import CFG
from pl_segformer_datamodule import SegFormerDataModule
from util.config_handler import save_config, load_config
from util.train_utils import get_device_configuration, get_data_root_dir
from pl_segformer_lightning import SegFormerLightningModule
torch.set_float32_matmul_precision('medium')


def main():
    cfg = CFG()

    seed_value = cfg.seed
    seed_everything(seed_value)

    wandb_logger = WandbLogger(project="Kaggle1stReimp", entity="wuesuv")
    wandb_run_name = wandb_logger.experiment.name
    timestamp = datetime.now().strftime("%y%m%d-%H%M")  # Format: YYMMDD-HHMM

    model_run_dir = os.path.join("checkpoints", f"{timestamp}-{wandb_run_name}")
    save_config(config=cfg, model_run_dir=model_run_dir)

    model = SegFormerLightningModule()
    data_module = SegFormerDataModule(data_root_dir=get_data_root_dir(cfg))

    checkpoint_callback = ModelCheckpoint(
        dirpath=f"checkpoints/{timestamp}-{wandb_run_name}",
        filename="best-checkpoint-{epoch}-{val_iou:.2f}--{val_auc:.2f}",
        save_top_k=1,
        monitor="val_iou",
        mode="max",
        every_n_epochs=1,
    )

    devices = get_device_configuration()
    print("Using Devices:", devices)

    trainer = Trainer(
        max_epochs=cfg.epochs,
        logger=wandb_logger,
        callbacks=[LearningRateMonitor(logging_interval='epoch'), checkpoint_callback],
        accelerator="auto",
        devices=devices,
        enable_progress_bar=True,
    )

    trainer.fit(model, data_module)


if __name__ == '__main__':
    main()
