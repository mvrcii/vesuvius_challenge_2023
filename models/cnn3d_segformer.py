import torch
import torch.nn as nn
from transformers import SegformerForSemanticSegmentation, SegformerConfig


class CNN3D_Segformer(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv3d_1 = nn.Conv3d(1, 4, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1))
        self.conv3d_2 = nn.Conv3d(4, 8, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1))
        self.conv3d_3 = nn.Conv3d(8, 16, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1))
        self.conv3d_4 = nn.Conv3d(16, 32, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1))

        ckpt_path = "nvidia/segformer-b1-finetuned-ade-512-512"
        self.xy_encoder_2d = SegformerForSemanticSegmentation.from_pretrained(ckpt_path,
                                                                              config=cnn_3d_segformer_b1_config,
                                                                              ignore_mismatched_sizes=True)

    def forward(self, image):
        output = self.conv3d_1(image)
        output = self.conv3d_2(output)
        output = self.conv3d_3(output)
        output = self.conv3d_4(output).max(axis=2)[0]
        output = self.xy_encoder_2d(output).logits

        output = output.squeeze(1)
        output = torch.sigmoid(output)
        return output


cnn_3d_segformer_b1_config = SegformerConfig(
    **{
        "architectures": ["SegformerForImageClassification"],
        "attention_probs_dropout_prob": 0.0,
        "classifier_dropout_prob": 0.1,
        "decoder_hidden_size": 256,
        "depths": [2, 2, 2, 2],
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
