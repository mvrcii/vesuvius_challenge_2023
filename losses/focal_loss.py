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


class MaskedFocalLoss(nn.Module):

    def __init__(self, gamma=0, alpha=0.25):
        super(MaskedFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, y_pred_unmasked, y_true_unmasked, y_mask):
        """
        :param y_pred_unmasked: Logits [N, H, W]
        :param y_true_unmasked: [N, H, W]
        :param y_mask: [N, H, W]
        :return:
        """
        y_pred_unmasked = y_pred_unmasked.squeeze()

        y_pred = y_pred_unmasked * y_mask
        y_true = y_true_unmasked * y_mask

        return torchvision.ops.sigmoid_focal_loss(y_pred, y_true, self.alpha, self.gamma, reduction='mean')
