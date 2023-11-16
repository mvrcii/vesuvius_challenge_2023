import torch
import torch.nn as nn
from transformers import SegformerForSemanticSegmentation, SegformerConfig

from unetr import UNETR


class UNETR_Segformer(nn.Module):
    def __init__(self, cfg, dropout=.2):
        super().__init__()
        self.cfg = cfg
        self.dropout = nn.Dropout2d(dropout)
        self.encoder = UNETR(
            input_dim=1,
            output_dim=32,
            img_shape=(16, self.cfg.patch_size, self.cfg.patch_size)
        )

        self.encoder_2d = SegformerForSemanticSegmentation(unet_3d_jumbo_config)
        self.upscaler1 = nn.ConvTranspose2d(
            1, 1, kernel_size=(4, 4), stride=2, padding=1)
        self.upscaler2 = nn.ConvTranspose2d(
            1, 1, kernel_size=(4, 4), stride=2, padding=1)

        # self.batch_norm = nn.BatchNorm2d(16) <- 16 = number of feature maps/channels (applied channel-wise)
        # self.batch_norm_upscale1 = nn.BatchNorm2d(..)
        # self.batch_norm_upscale2 = nn.BatchNorm2d(..)

    def forward(self, image):
        output = self.encoder(image).max(axis=2)[0]  # 512, 512, 16 -> 512, 512, 16 (16 channels/feature maps??)
        # TODO: Add BatchNorm2d/3D
        # TODO: output = self.batch_norm(output)
        output = self.dropout(output)
        output = self.encoder_2d(output).logits
        output = self.upscaler1(output)
        # TODO: Add BatchNorm2d/3D
        # TODO: output = self.batch_norm_upscale1(output)
        output = self.upscaler2(output)
        # TODO: Add BatchNorm2d/3D
        # TODO: output = self.batch_norm_upscale2(output)

        output = output.squeeze(1)
        output = torch.sigmoid(output)

        return output


unet_3d_jumbo_config = SegformerConfig(
    **{
        "architectures": ["SegformerForImageClassification"],
        "attention_probs_dropout_prob": 0.0,
        "classifier_dropout_prob": 0.1,
        "decoder_hidden_size": 768,
        "depths": [3, 6, 40, 3],
        "downsampling_rates": [1, 4, 8, 16],
        "drop_path_rate": 0.1,
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.0,
        "hidden_sizes": [64, 128, 320, 512],
        "image_size": 224,
        "initializer_range": 0.02,
        "layer_norm_eps": 1e-06,
        "mlp_ratios": [4, 4, 4, 4],
        "model_type": "segformer",
        "num_attention_heads": [1, 2, 5, 8],
        "num_channels": 32,
        "num_encoder_blocks": 4,
        "patch_sizes": [7, 3, 3, 3],
        "sr_ratios": [8, 4, 2, 1],
        "strides": [4, 2, 2, 2],
        "torch_dtype": "float32",
        "transformers_version": "4.12.0.dev0",
        "num_labels": 1,
    }
)
