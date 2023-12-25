import argparse
import os
import pickle
import sys
import types
import warnings
from datetime import datetime

import numpy as np
import torch
from lightning import seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.trainer import Trainer
from lightning_fabric.accelerators import find_usable_cuda_devices

from models.data_modules.segformer_datamodule import SegFormerDataModule
from models.data_modules.unet3d_datamodule import UNET3D_DataModule
from models.data_modules.unet3dsf_datamodule import UNET3D_SFDataModule
from models.data_modules.unetrsf_datamodule import UNETR_SFDataModule
from models.lightning_modules.segformer_module import SegformerModule
from models.lightning_modules.unet3d_module import UNET3D_Module
from models.lightning_modules.unet3dsf_module import UNET3D_SFModule
from models.lightning_modules.unetrsf_module import UNETR_SFModule
from models.lightning_modules.vit3d_module import Vit3D_Module
from utility.configs import Config

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
    elif architecture == 'unetr-sf':
        return UNETR_SFModule(cfg=config)
    elif architecture == 'unet3d-sf':
        return UNET3D_SFModule(cfg=config)
    elif architecture == 'unet3d':
        return UNET3D_Module(cfg=config)
    elif architecture == 'vit3d':
        return Vit3D_Module(cfg=config)
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

    if architecture == "segformer":
        return SegFormerDataModule(cfg=config)
    elif architecture == "unetr-sf":
        return UNETR_SFDataModule(cfg=config)
    elif architecture == "unet3d-sf":
        return UNET3D_SFDataModule(cfg=config)
    elif architecture == 'unet3d' or architecture == 'vit3d':
        return UNET3D_DataModule(cfg=config)
    else:
        print("Invalid architecture for data module:", architecture)
        sys.exit(1)


def get_device_configuration():
    """
    Determines the appropriate device configuration for training based on
    the availability of CUDA-enabled GPUs.

    :return: A tuple (accelerator, devices) where:
        - 'accelerator' is a string indicating the type of accelerator ('gpu' or 'cpu').
        - 'devices' is an int or list indicating the devices to be used.
    """
    if torch.cuda.is_available():
        # Return all available GPUs
        gpu_ids = find_usable_cuda_devices()
        return gpu_ids
    else:
        # No GPUs available, use CPU
        return 1


def get_callbacks(cfg, model_run_dir):
    node = getattr(cfg, 'node', True)  # defaults to true

    # Only save model checkpoint on gpu node
    if node:
        os.makedirs(model_run_dir, exist_ok=True)
        cfg.save_to_file(model_run_dir)

        monitor_metric = "val_auc"
        if cfg.architecture == "unet3d":
            monitor_metric = "val_iou"

        checkpoint_callback = ModelCheckpoint(
            dirpath=model_run_dir,
            filename="best-checkpoint-{epoch}-{val_iou:.2f}",
            save_top_k=1,
            monitor=monitor_metric,
            mode="max",
            every_n_epochs=1
        )

        return [checkpoint_callback]
    else:
        return []


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
    model_name = getattr(config, 'model_name', 'default_model')
    wandb_generated_name = wandb_logger.experiment.name
    model_run_name = f"{wandb_generated_name}-{model_name}-{timestamp}"
    wandb_logger.experiment.name = model_run_name
    model_run_dir = os.path.join(config.work_dir, "checkpoints", model_run_name)

    model = get_model(config=config)
    data_module = get_data_module(config=config)

    trainer = Trainer(
        max_epochs=config.epochs,
        logger=wandb_logger,
        callbacks=get_callbacks(cfg=config, model_run_dir=model_run_dir),
        accelerator="auto",
        devices=get_device_configuration(),
        enable_progress_bar=True,
        precision='16-mixed',
        gradient_clip_val=1.0,
        gradient_clip_algorithm="norm",
        check_val_every_n_epoch=config.val_interval
    )

    trainer.fit(model, data_module)


if __name__ == '__main__':
    main()
