from torch.nn import BCEWithLogitsLoss

from losses.binary_dice_loss import BinaryDiceLoss


def get_loss_functions(cfg):
    available_losses = {"dice": BinaryDiceLoss(from_logits=True),
                        "bce": BCEWithLogitsLoss()}

    return [(name, weight, available_losses[name]) for (name, weight) in cfg.losses]
