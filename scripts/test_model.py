import torch
from transformers import SegformerForSemanticSegmentation

model = SegformerForSemanticSegmentation.from_pretrained(
    pretrained_model_name_or_path="nvidia/mit-b2",
    ignore_mismatched_sizes=True,
    num_labels=1,
    num_channels=1,
)

sample_input = torch.rand((1, 1, 512, 512))

output = model(sample_input)

print(output)
