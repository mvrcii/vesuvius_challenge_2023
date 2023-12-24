import os

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from mmseg.apis import inference_model
from mmseg.apis import init_model

'''
        SET WORKING DIRECTORY TO PROJECT ROOT
        
        Similarly to infer_full_image, this script runs inference on patches across an entire image.
        However overlap_factor can be set to have overlapping patches, it works by directly decreasing 
        the stride. Overlap 2 => Stride = patch_size / 2, 
        Bigger overlap_factor => Smaller strides => More overlap
        Bigger Overlap Factor => Longer inference time, recommended range: [2 .. 5]
'''

# Load the .tif image
overlap_factor = 2
patch_size = 512
stride = int(patch_size / overlap_factor)

img_path = os.path.join('data', 'fragments', 'fragment2', 'slices', '00032.tif')
large_img = Image.open(img_path)
width, height = large_img.size

config_file = 'mmseg_test/work_dirs/segformer_VesuvDataset/segformer_mit-b5_512x512_160k_ade20k.py'
checkpoint_file = 'mmseg_test/work_dirs/segformer_VesuvDataset/best_aAcc_epoch_11.pth'
model = init_model(config_file, checkpoint_file, device='cuda:0')

large_img = (np.asarray(large_img) / 65535.0).astype(np.float16)
large_img = (large_img * 255).round().clip(0, 255).astype(np.uint8)  # Ensure values are within 0-255

# Define the size of the patches
height, width = large_img.shape

# Calculate the number of patches needed, ensuring coverage of the entire image
x_patches = int(np.ceil(width / patch_size))
y_patches = int(np.ceil(height / patch_size))

total = (x_patches * overlap_factor) * (y_patches * overlap_factor)
done = 0

# Initialize an empty array to hold the stitched result
stitched_result = np.zeros((y_patches * patch_size, x_patches * patch_size), dtype=np.float32)
vote_count = np.zeros((height, width), dtype=np.int_)

# Process each patch
for y in range(0, height - patch_size + 1, stride):
    for x in range(0, width - patch_size + 1, stride):
        print(f"{done}/{total}")
        patch = large_img[y:y + patch_size, x:x + patch_size]

        # If the patch size is smaller than the expected size, pad it
        if patch.shape < (patch_size, patch_size):
            patch = np.pad(patch, ((0, patch_size - patch.shape[0]), (0, patch_size - patch.shape[1])), 'constant')

        # Perform inference on the patch
        result = inference_model(model, patch)

        # Convert result to numpy array if it's a tensor
        result_np = result.pred_sem_seg.data.squeeze(0).cpu().numpy()

        # Place the result in the corresponding location in the stitched_result array and count votes
        stitched_result[y:y + patch_size, x:x + patch_size] += result_np
        vote_count[y:y + patch_size, x:x + patch_size] += 1

        done += 1

# Crop the stitched_result to the original image size
stitched_result = stitched_result[:height, :width]

stitched_result = stitched_result / vote_count  # Average the results where patches overlap
majority_result = np.round(stitched_result)  # Majority vote

# Convert the stitched result to an image and display it
plt.imshow(majority_result, cmap='gray')
plt.axis('off')  # Hide the axis
plt.show()
