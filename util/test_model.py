import torch
from torch import float16
from transformers import SegformerForSemanticSegmentation

model = SegformerForSemanticSegmentation.from_pretrained(
    pretrained_model_name_or_path="nvidia/mit-b2",
    ignore_mismatched_sizes=True,
    num_labels=1,
    num_channels=1,
)

model = model.to('cuda')
model = model.half()

sample_input = torch.rand((2, 1, 512, 512), device='cuda', dtype=float16)

output = model(sample_input)

print(output)
