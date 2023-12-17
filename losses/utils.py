def get_loss_functions(cfg):
    available_losses = {"dice": BinaryDiceLoss(from_logits=True),
                        "bce": BCEWithLogitsLoss()}

    return [(name, weight, available_losses[name]) for (name, weight) in cfg.losses]
