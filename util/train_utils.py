import os
import warnings

import torch
from lightning_fabric.accelerators import find_usable_cuda_devices


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


def get_data_root_dir(config):
    if config.k_fold:
        print("Start training with k_fold data")
        train_ids_str, val_ids_str = build_k_fold_folder(config.train_frag_ids, config.val_frag_ids)
        data_root_dir = os.path.join(config.data_root_dir, f'k_fold_{train_ids_str}_{val_ids_str}', str(config.size))
    else:
        print("Start training with single fragment data")
        data_root_dir = os.path.join(config.data_root_dir, f'single_TF{config.single_train_frag_id}', str(config.size))

    return data_root_dir


def build_k_fold_folder(train_frag_ids, val_frag_ids):
    train_ids_formatted = ['TF{}'.format(idx) for idx in train_frag_ids]
    val_ids_formatted = ['VF{}'.format(idx) for idx in val_frag_ids]
    train_ids_str = '_'.join(train_ids_formatted)
    val_ids_str = '_'.join(val_ids_formatted)

    return train_ids_str, val_ids_str


def load_config(config):
    """
    Load local configuration from conf_local.py if available.
    Overrides the default configurations in CFG with values found in conf_local.py.
    If conf_local.py is not found, a warning is issued and default configurations are used.

    :return: None
    """
    try:
        import conf_local
        for key in dir(conf_local):
            if not key.startswith("__"):  # Exclude built-in attributes
                # Set attribute if it exists in CFG
                if hasattr(config, key):
                    setattr(config, key, getattr(conf_local, key))
    except ImportError:
        warnings.warn("Local configuration file 'conf_local.py' not found. Using default configuration.",
                      RuntimeWarning)

    return config
