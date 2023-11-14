import matplotlib.pyplot as plt
import numpy as np
import torch
import random
import os
from transformers import SegformerForSemanticSegmentation

ckpt_path = "../checkpoints/231114-0143-ethereal-sun-162/best-checkpoint-epoch=19-val_iou=0.97.ckpt"

# Assuming your images and labels are stored in these directories
image_dir = "../data/512/train/images/"
label_dir = "../data/512/train/labels/"

# Randomly select 5 image-label pairs
image_files = os.listdir(image_dir)
selected_files = random.sample(image_files, 5)

# Load your model
model = SegformerForSemanticSegmentation.from_pretrained("nvidia/mit-b3", num_labels=1, num_channels=16, ignore_mismatched_sizes=True)
checkpoint = torch.load(ckpt_path)

state_dict = {key.replace('model.', ''): value for key, value in checkpoint['state_dict'].items()}
model.load_state_dict(state_dict)

# Set up the figure for plotting
fig, axs = plt.subplots(5, 3, figsize=(10, 10))  # 5 rows for images, 3 columns for logits, binary, and label

for i, file in enumerate(selected_files):
    input_tensor = np.load(image_dir + file)
    label_tensor = np.load(label_dir + file.replace('images', 'labels'))

    outputs = model(torch.tensor(input_tensor).unsqueeze(0).float())
    logits = outputs.logits
    logits = logits.detach().squeeze().cpu()

    # Plot each of the three images for this sample
    axs[i, 0].imshow(logits, cmap='gray')

    axs[i, 1].imshow(logits > 0.5, cmap='gray')

    axs[i, 2].imshow(label_tensor, cmap='gray')

    if i == 0:
        axs[i, 0].title.set_text('logits')
        axs[i, 1].title.set_text('binary')
        axs[i, 2].title.set_text('label')

plt.tight_layout()
plt.show()
