from transformers import SegformerForSemanticSegmentation
import numpy as np
import torch
import matplotlib.pyplot as plt
import torch
from label_helper import make_label

label_tensor = make_label()
input_tensor = torch.stack([label_tensor] * 16, dim=1)

print(input_tensor.shape)
exit()

# define model
model = SegformerForSemanticSegmentation.from_pretrained("nvidia/mit-b0",
                                                         num_labels=1,
                                                         num_channels=16,
                                                         ignore_mismatched_sizes=True)

optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
# print(model.config)

for i in range(5):
    optimizer.zero_grad()
    outputs = model(pixel_values=input_tensor, labels=label_tensor)
    loss, logits = outputs.loss, outputs.logits
    print(loss)
    loss.backward()
    optimizer.step()
    test = (logits > 0.5).int()
plt.imshow(test.detach().squeeze().cpu().numpy(), cmap='gray')
# plt.imshow(logits.detach().squeeze().cpu().numpy(), cmap='gray')
plt.show()
