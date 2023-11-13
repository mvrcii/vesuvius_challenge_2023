import torch.nn as nn
from transformers import SegformerForSemanticSegmentation


class MultiChannelSegformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.segformer = SegformerForSemanticSegmentation.from_pretrained("nvidia/mit-b5",
                                                                          num_labels=1,
                                                                          num_channels=16,
                                                                          ignore_mismatched_sizes=True,
                                                                          )
        self.upscaler1 = nn.ConvTranspose2d(
            1, 1, kernel_size=(4, 4), stride=2, padding=1)
        self.upscaler2 = nn.ConvTranspose2d(
            1, 1, kernel_size=(4, 4), stride=2, padding=1)

    def forward(self, image):
        output = self.segformer(image).logits
        output = self.upscaler1(output)
        output = self.upscaler2(output)
        output = output.squeeze(1)
        # output = torch.sigmoid(output)
        return output
