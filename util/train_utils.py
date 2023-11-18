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


