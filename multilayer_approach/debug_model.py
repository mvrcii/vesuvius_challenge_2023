import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy.compat import long
from skimage.transform import resize
from torch import float16
from torch.autograd import Variable
from tqdm import tqdm

from scripts.test.testloss import dice_loss_with_mask, binary_cross_entropy_with_mask
from models.architectures.unetr_segformer import UNETR_Segformer
from utility.configs import Config


class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int, long)): self.alpha = torch.Tensor([alpha, 1 - alpha])
        if isinstance(alpha, list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim() > 2:
            input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1, 2)  # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))  # N,H*W,C => N*H*W,C
        target = target.view(-1, 1)

        logpt = F.log_softmax(input)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1 - pt) ** self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()


def create_data(cfg):
    label = np.zeros((cfg.patch_size, cfg.patch_size))
    start = (cfg.patch_size // 2) - 64
    end = (cfg.patch_size // 2) + 64
    label[start+32:end+32, start:end] = 1

    image = np.stack([label] * 16)

    label = resize(label, ((cfg.patch_size // 4), (cfg.patch_size // 4)))

    return image, label


if __name__ == "__main__":

    cfg = Config.load_from_file("configs/unet_sf/config_debug.py")

    model = UNETR_Segformer(cfg)

    optimizer = torch.optim.AdamW(params=model.parameters(), lr=0.00001)
    bce_loss_fn = torch.nn.BCEWithLogitsLoss()
    focal_loss_fn = FocalLoss()

    if torch.cuda.is_available():
        model.to('cuda')
    else:
        print("CUDA is not available. The model will remain on the CPU.")

    image, label = create_data(cfg)

    mask_np = np.ones_like(label)
    # mask_np[:, 20:40] = 0
    # plt.imshow(mask_np, cmap='gray')
    # plt.show()

    mask = torch.tensor(mask_np, dtype=float16).to('cuda')
    label = torch.tensor(label, dtype=float16).to('cuda')

    # plt.imshow(label, cmap='gray')
    # plt.imshow(image, cmap='gray')
    # plt.show()

    image = torch.Tensor(image).unsqueeze(0).unsqueeze(0).float()
    assert image.dim() == 5
    print("Image Shape:", image.shape)

    image = image.to('cuda')

    epochs = 200
    for x in tqdm(range(epochs)):
        # Forward Pass
        logits = model(image)
        logits = logits.half()

        probabilities = torch.sigmoid(logits)

        bce_loss = binary_cross_entropy_with_mask(logits, label.unsqueeze(0), mask)
        dice_loss = dice_loss_with_mask(probabilities, label, mask)
        total_loss = bce_loss + dice_loss

        optimizer.zero_grad()
        dice_loss.backward()
        optimizer.step()

        print("Loss:", dice_loss)

        prediction = probabilities.squeeze().detach().cpu().numpy()
        plt.imshow(prediction, cmap='gray')
        plt.show()
