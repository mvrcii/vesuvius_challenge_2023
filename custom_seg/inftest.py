from transformers import SegformerForSemanticSegmentation
import numpy as np
import torch
import matplotlib.pyplot as plt
import torch


label_tensor = np.load("../data/512/train/labels/4096_9984_4608_10496.npy")
input_tensor = np.load("../data/512/train/images/4096_9984_4608_10496.npy")
input_tensor =torch.tensor(input_tensor.transpose(2, 0, 1)).unsqueeze(0).float()
input_tensor =  input_tensor / 255.0

# plt.imshow(label_tensor)
print(label_tensor.shape)
print(input_tensor.shape)
# plt.show()
# exit()



# define model
model = SegformerForSemanticSegmentation.from_pretrained("nvidia/mit-b0",
                                                         num_labels=1,
                                                         num_channels=16,
                                                         ignore_mismatched_sizes=True)

checkpoint = torch.load("../model_512_0.01_epoch_10.pth")

model.load_state_dict(checkpoint)

outputs = model(input_tensor)

logits = outputs.logits
logits = logits.detach().squeeze().cpu()

# plt.imshow(logits.detach().squeeze().cpu(), cmap='gray')
# plt.show()

plt.figure(figsize=(20, 10))

# Loop to create 10 subplots with different thresholds
for i in range(10):
    threshold = i * 0.05 - 1.6 # Thresholds from 0 to 0.5
    binary_image = logits > threshold  # Convert to binary using the threshold

    plt.subplot(2, 5, i + 1)  # 2 rows, 5 columns
    plt.imshow(binary_image, cmap='gray')
    plt.title(f'Threshold: {threshold:.2f}')
    plt.axis('off')

plt.tight_layout()
plt.show()