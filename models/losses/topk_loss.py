import numpy as np
from torch import nn
from wandb.wandb_torch import torch


class TopKLoss(nn.BCEWithLogitsLoss):
    """
    TopKLoss for 2d binary segmentation
    Requires specifying percentage of pixels kept
    Expects input and target of same shape, target can be int or float
    pos_weight (Tensor, optional) – a weight of positive examples. Must be a vector
    with length equal to the number of classes. A multiplier for class frequencies.

    l_bce = nn.BCEWithLogitsLoss()
    l_top10 = TopKLoss(k=10)
    l_top100 = TopKLoss(k=100)

    print(l_top10(input.squeeze(), target))
    print(l_bce(input.squeeze(), target))
    print(l_top100(input.squeeze(), target))

    See https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunet/training/loss_functions/TopK_loss.py
    for a more general implementation (multi-class, 3d)
    """

    def __init__(self, pos_weight=None, k=10):
        self.k = k
        super(TopKLoss, self).__init__(reduction='none', pos_weight=pos_weight)

    def forward(self, inp, target):
        res = super(TopKLoss, self).forward(inp, target)
        num_pixels = np.prod(res.shape, dtype=np.int64)
        res, _ = torch.topk(res.view((-1,)), int(num_pixels * self.k / 100), sorted=False)
        return res.mean()
