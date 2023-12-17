import torchvision
from torch import nn


class FocalLoss(nn.Module):
    """
    inputs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.25, (optional) weighting factor in range (0,1) to balance positive vs negative examples or -1 for ignore. Default = 0.25
    gamma: float = 2,  exponent of the modulating factor (1 - p_t) to balance easy vs hard examples.
    reduction: str = 'mean'

    See https://amaarora.github.io/2020/06/29/FocalLoss.html for an explanation
    """

    def __init__(self, alpha=0.25, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inp, target):
        return torchvision.ops.sigmoid_focal_loss(inp, target, self.alpha, self.gamma, self.reduction)
