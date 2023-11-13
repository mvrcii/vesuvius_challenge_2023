import torch.nn as nn
from transformers import SegformerForSemanticSegmentation


class MultiChannelSegformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.segformer = SegformerForSemanticSegmentation.from_pretrained("nvidia/mit-b0",
                                                                          num_labels=1,
                                                                          num_channels=16,
                                                                          ignore_mismatched_sizes=True,
                                                                          )
        self.upscaler1 = nn.ConvTranspose2d(
            1, 1, kernel_size=(4, 4), stride=2, padding=1)
        self.upscaler2 = nn.ConvTranspose2d(
            1, 1, kernel_size=(4, 4), stride=2, padding=1)

    def forward(self, pixel_values, labels):
        output_tuple = self.segformer(pixel_values=pixel_values, labels=labels)
        logits, loss = output_tuple.logits, output_tuple.loss

        output = self.upscaler1(logits)
        output = self.upscaler2(output)

        output = output.squeeze(1)
        # output = torch.sigmoid(output)
        return output, loss
