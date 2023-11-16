from transformers import SegformerForSemanticSegmentation
import torch

from conf import CFG

model = SegformerForSemanticSegmentation.from_pretrained(CFG.seg_pretrained,
                                                         num_labels=1,
                                                         num_channels=64,
                                                         ignore_mismatched_sizes=True)

# torch.save(model.state_dict(), "b364channel")


def print_model_parameters(model):
    total_params = 0
    print("Model Parameters:")
    for name, param in model.named_parameters():
        print(f"Layer: {name} | Size: {param.patch_size()} | Count: {param.nelement()}")
        total_params += param.nelement()
    print(f"Total number of parameters: {total_params}")


print_model_parameters(model)
