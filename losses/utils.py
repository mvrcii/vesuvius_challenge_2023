from torch.nn import BCEWithLogitsLoss

from losses.binary_dice_loss import BinaryDiceLoss
from losses.entropy_loss import EntropyLoss
from losses.focal_loss import FocalLoss


def get_loss_functions(cfg):
    available_losses = {"dice": BinaryDiceLoss(from_logits=True),
                        "bce": BCEWithLogitsLoss(),
                        "focal": FocalLoss(gamma=3.0, alpha=0.20),
                        "entropy": EntropyLoss()}

    return [(name, weight, available_losses[name]) for (name, weight) in cfg.losses]
