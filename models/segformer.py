import os

import torch
from transformers import SegformerForSemanticSegmentation

from models.abstract_model import AbstractVesuvLightningModule


class SegformerModule(AbstractVesuvLightningModule):
    def __init__(self, cfg):
        super().__init__(cfg=cfg)

        self.model = SegformerForSemanticSegmentation.from_pretrained(
            pretrained_model_name_or_path=cfg.from_pretrained,
            ignore_mismatched_sizes=True,
            num_labels=1,
            num_channels=cfg.in_chans,
        )

        if cfg.from_checkpoint:
            checkpoint_root_path = os.path.join("checkpoints", cfg.from_checkpoint)
            checkpoint_files = [file for file in os.listdir(checkpoint_root_path) if file.startswith('best-checkpoint')]
            checkpoint_path = os.path.join(checkpoint_root_path, checkpoint_files[-1])
            print(checkpoint_path)
            checkpoint = torch.load(checkpoint_path)
            assert checkpoint
            state_dict = {key.replace('model.', ''): value for key, value in checkpoint['state_dict'].items()}
            self.model.load_state_dict(state_dict)
            print("Loaded model from checkpoint:", cfg.from_checkpoint)
        else:
            print("Loaded model from pretrained:", cfg.from_pretrained)
