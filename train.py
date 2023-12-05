import argparse
import os
import sys
import types
import warnings
from datetime import datetime

import numpy as np
import torch
from lightning import seed_everything, Callback
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.trainer import Trainer

from config_handler import Config
from data_modules.segformer import SegFormerDataModule
from data_modules.simplecnn import SimpleCNNDataModule
from models.cnn3d_segformer import CNN3D_SegformerModule
from models.segformer import SegformerModule
from models.simplecnn import SimpleCNNModule
from models.unetplusplus import UnetPlusPlusModule
from util.train_utils import get_device_configuration

torch.set_float32_matmul_precision('medium')

warnings.filterwarnings("ignore",
                        message="Some weights * were not initialized from the model checkpoint")
warnings.filterwarnings("ignore",
                        message="You should probably TRAIN this model on a down-stream task")

warnings.simplefilter("ignore", category=Warning)


def get_sys_args():
    parser = argparse.ArgumentParser(description='Train configuration.')

    parser.add_argument('config_path', type=str, help='Path to the configuration file')
    parser.add_argument('--seed', type=int, default=None, help='Optional seed for the script')

    # Parsing the arguments
    args = parser.parse_args()

    return args


def get_model(config: Config):
    architecture = config.architecture

    if architecture == 'segformer':
        return SegformerModule(cfg=config)
    elif architecture == 'cnn3d_segformer':
        return CNN3D_SegformerModule(cfg=config)
    elif architecture == 'unetplusplus':
        return UnetPlusPlusModule(cfg=config)
    elif architecture == 'simplecnn':
        return SimpleCNNModule(cfg=config)
    else:
        print("Invalid architecture for model:", architecture)
        sys.exit(1)


def log_wandb_hyperparams(config, wandb_logger):
    config_dict = vars(config)  # Convert config object to a dictionary

    # Remove non-serializable items
    cleaned_config = {k: v for k, v in config_dict.items() if not isinstance(v, types.ModuleType)}

    # Log the cleaned hyperparameters
    wandb_logger.log_hyperparams(cleaned_config)


def get_data_module(config: Config):
    architecture = config.architecture

    if architecture == "segformer" or architecture == "unetplusplus":
        return SegFormerDataModule(cfg=config)
    # elif architecture == "unetplusplus":
    #     return UnetPlusPlusDataModule(cfg=config)
    elif architecture == "simplecnn":
        return SimpleCNNDataModule(cfg=config)
    else:
        print("Invalid architecture for data module:", architecture)
        sys.exit(1)


class NaNStoppingCallback(Callback):
    def __init__(self, logger):
        super().__init__()
        self.logger = logger

    def on_validation_end(self, trainer, pl_module):
        val_loss = trainer.callback_metrics.get('val_loss')
        val_iou = trainer.callback_metrics.get('val_iou')
        if val_loss is not None and torch.isnan(val_loss).any():
            trainer.should_stop = True
            print(f"Halting training due to NaN in validation loss at epoch {trainer.current_epoch}")
            # Use the logger passed to the callback
            self.logger.experiment.log({'halted': True, 'reason': 'NaN in val_loss', 'epoch': trainer.current_epoch})
        if val_iou is not None and torch.isnan(val_iou).any():
            trainer.should_stop = True
            print(f"Halting training due to NaN in validation IoU at epoch {trainer.current_epoch}")
            # Use the logger passed to the callback
            self.logger.experiment.log({'halted': True, 'reason': 'NaN in val_iou', 'epoch': trainer.current_epoch})


def main():
    args = get_sys_args()

    config = Config.load_from_file(args.config_path)

    if args.seed:
        seed_everything(args.seed)
        np.random.seed(args.seed)
    elif config.seed != -1:
        seed_everything(config.seed)
        np.random.seed(config.seed)

    wandb_logger = WandbLogger(project="Kaggle1stReimp", entity="wuesuv")

    log_wandb_hyperparams(config=config, wandb_logger=wandb_logger)

    # Model name related stuff
    timestamp = datetime.now().strftime('%y%m%d-%H%M%S')
    model_name = config.model_name if hasattr(config, 'model_name') else "default_model"
    wandb_generated_name = wandb_logger.experiment.name
    model_run_name = f"{wandb_generated_name}-{model_name}-{timestamp}"
    wandb_logger.experiment.name = model_run_name
    model_run_dir = os.path.join(config.work_dir, "checkpoints", model_run_name)

    model = get_model(config=config)

    data_module = get_data_module(config=config)

    checkpoint_callback = ModelCheckpoint(
        dirpath=model_run_dir,
        filename="best-checkpoint-{epoch}-{val_iou:.2f}",
        save_top_k=1,
        monitor="val_iou",
        mode="max",
        every_n_epochs=1
    )

    nan_stopping_callback = NaNStoppingCallback(logger=wandb_logger)

    devices = get_device_configuration()
    print("Using Devices:", devices)

    trainer = Trainer(
        max_epochs=config.epochs,
        logger=wandb_logger,
        callbacks=[checkpoint_callback, nan_stopping_callback],
        accelerator="auto",
        devices=devices,
        enable_progress_bar=True,
        precision='16-mixed',
        gradient_clip_val=1.0,
        gradient_clip_algorithm="norm",
        check_val_every_n_epoch=config.val_interval
    )

    os.makedirs(model_run_dir, exist_ok=True)
    config.save_to_file(model_run_dir)

    trainer.fit(model, data_module)


if __name__ == '__main__':
    main()
