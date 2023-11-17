from datetime import datetime

from lightning import seed_everything
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.trainer import Trainer

from conf import CFG
from pl_segformer_datamodule import SegFormerDataModule
from pl_segformer_lightning import SegFormerLightningModule
from util.train_utils import get_device_configuration, get_data_root_dir, load_config


def main():
    cfg = load_config(CFG)

    seed_value = CFG.seed
    seed_everything(seed_value)

    wandb_logger = WandbLogger(project="Kaggle1stReimp", entity="wuesuv")
    wandb_run_name = wandb_logger.experiment.name
    timestamp = datetime.now().strftime("%y%m%d-%H%M")  # Format: YYMMDD-HHMM

    model = SegFormerLightningModule()
    data_module = SegFormerDataModule(data_root_dir=get_data_root_dir(cfg))

    checkpoint_callback = ModelCheckpoint(
        dirpath=f"checkpoints/{timestamp}-{wandb_run_name}",
        filename="best-checkpoint-{epoch}-{val_iou:.2f}",
        save_top_k=1,
        monitor="val_iou",
        mode="max",
        every_n_epochs=1,
    )

    devices = get_device_configuration()
    print("Using Devices:", devices)

    trainer = Trainer(
        max_epochs=CFG.epochs,
        logger=wandb_logger,
        callbacks=[LearningRateMonitor(logging_interval='epoch'), checkpoint_callback],
        accelerator="auto",
        devices=devices,
        enable_progress_bar=True,
        # log_every_n_steps=5,
    )

    trainer.fit(model, data_module)


if __name__ == '__main__':
    main()
