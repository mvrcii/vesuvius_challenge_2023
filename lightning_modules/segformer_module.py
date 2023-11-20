from transformers import SegformerForSemanticSegmentation

from lightning_modules.abstract_module import AbstractVesuvLightningModule


class SegformerModule(AbstractVesuvLightningModule):
    def __init__(self, cfg):
        super().__init__(cfg=cfg)
        self.model = SegformerForSemanticSegmentation.from_pretrained(
            cfg.from_pretrained,
            num_labels=1,
            num_channels=cfg.in_chans,
            ignore_mismatched_sizes=True,
        )
