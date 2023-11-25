from transformers import SegformerForSemanticSegmentation, SegformerConfig

from models.abstract_module import AbstractVesuvLightningModule


class SegformerModule(AbstractVesuvLightningModule):
    def __init__(self, cfg):
        super().__init__(cfg=cfg)
        segformer_config = SegformerConfig(
            # hidden_dropout_prob=0.05,
            # drop_path_rate=0.05,
            # attention_probs_dropout_prob=0.05,
            # classifier_dropout_prob=0.05,
            num_labels=1,
            num_channels=cfg.in_chans,
        )

        self.model = SegformerForSemanticSegmentation.from_pretrained(
            pretrained_model_name_or_path=cfg.from_pretrained,
            ignore_mismatched_sizes=True,
            config=segformer_config
        )
