import torch
from torch import nn


class EntropyLoss(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, predictions, target):
        p_log_p = torch.where(predictions > 0, predictions * torch.log(predictions), torch.zeros_like(predictions))
        return -p_log_p.mean()
