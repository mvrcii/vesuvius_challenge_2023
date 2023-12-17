import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss2d(nn.Module):

    def __init__(self, gamma=0, alpha=None):
        super(FocalLoss2d, self).__init__()

        self.gamma = gamma
        self.alpha = alpha

    def forward(self, input, target):
        """
        :param input: [N, H, W]
        :param target: [H, W]
        :return:
        """
        target = target[:, 0, :, :].squeeze()
        bce_loss = F.binary_cross_entropy_with_logits(input, target.float())
        pt = torch.exp(-bce_loss)

        # If alpha is set, use it to balance the classes
        if self.alpha is not None:
            alpha = torch.tensor(self.alpha)
            # Apply alpha weighting
            alpha_factor = alpha[1] * target + alpha[0] * (1 - target)
            bce_loss = alpha_factor * bce_loss

        # compute the negative likelihood
        focal_loss = ((1 - pt) ** self.gamma) * bce_loss

        return focal_loss.mean()

