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
