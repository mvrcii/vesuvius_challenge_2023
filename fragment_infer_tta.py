import os
import sys
from datetime import datetime

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm
from transformers import SegformerForSemanticSegmentation

from conf import CFG

'''
        SET WORKING DIRECTORY TO PROJECT ROOT
        
        Runs inference on full image, by slicing it into patches.
'''


def read_fragment(fragment_id):
    images = []

    mid = 65 // 2

    start = mid - CFG.in_chans // 2
    end = mid + CFG.in_chans // 2
    idxs = range(start, end)

    for i in tqdm(idxs):
        img_path = os.path.join(CFG.fragment_root_dir, "fragments", f"fragment{fragment_id}", "slices", f"{i:05}.tif")

        image = cv2.imread(img_path, 0)
        assert 1 < np.asarray(image).max() <= 255, "Invalid image"

        pad0 = (CFG.tile_size - image.shape[0] % CFG.tile_size) % CFG.tile_size
        pad1 = (CFG.tile_size - image.shape[1] % CFG.tile_size) % CFG.tile_size

        image = np.pad(image, [(0, pad0), (0, pad1)], constant_values=0)

        images.append(image)
    images = np.stack(images, axis=0)

    # label_path = os.path.join(CFG.fragment_root_dir, "fragments/fragment2/inklabels.png")
    # label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
    # label = np.pad(label, [(0, pad0), (0, pad1)], mode='constant', constant_values=0)
    # label = (label / 255).astype(np.uint8)
    # assert set(np.unique(np.array(label))) == {0, 1}, "Invalid label"

    # return images, label
    return images


def infer_full_fragment(fragment_index, checkpoint_path):
    images = read_fragment(fragment_index)

    # Load your model
    model = SegformerForSemanticSegmentation.from_pretrained("nvidia/mit-b3", num_labels=1, num_channels=16,
                                                             ignore_mismatched_sizes=True)
    checkpoint = torch.load(checkpoint_path)
    state_dict = {key.replace('model.', ''): value for key, value in checkpoint['state_dict'].items()}
    model.load_state_dict(state_dict)
    print("Loaded model", checkpoint_path)
    # Define the size of the patches
    patch_size = 512
    patch_size_out = 128
    stride = patch_size // 2
    height, width = images[0].shape

    # Calculate the number of patches needed, considering the stride
    x_patches = int(np.ceil((width - patch_size) / stride)) + 1
    y_patches = int(np.ceil((height - patch_size) / stride)) + 1

    # Initialize arrays to hold the stitched result and count of predictions for each pixel
    stitched_height = y_patches * stride // patch_size * patch_size_out
    stitched_width = x_patches * stride // patch_size * patch_size_out
    out_arr = np.zeros((stitched_height, stitched_width), dtype=np.float32)
    pred_counts = np.zeros((stitched_height, stitched_width), dtype=np.int32)

    progress_bar = tqdm(total=x_patches * y_patches, desc="Infer Full Fragment: Processing patches")

    for y in range(y_patches):
        for x in range(x_patches):
            progress_bar.update(1)

            x_start = x * stride
            x_end = min(x_start + patch_size, width)
            y_start = y * stride
            y_end = min(y_start + patch_size, height)

            patch = images[:, y_start:y_end, x_start:x_end]

            # If the patch size is smaller than the expected size, pad it
            if patch.shape[1] < patch_size or patch.shape[2] < patch_size:
                # Calculate padding for height and width
                padding_height = max(patch_size - patch.shape[1], 0)
                padding_width = max(patch_size - patch.shape[2], 0)

                # Apply padding
                patch = np.pad(patch, ((0, 0), (0, padding_height), (0, padding_width)), 'constant')

            # Perform inference on the patch
            outputs = model(torch.tensor(patch).unsqueeze(0).float())
            logits = outputs.logits
            logits_np = logits.detach().squeeze().cpu().numpy()

            # Calculate the margin to ignore (10% of the patch size)
            margin = int(0.1 * patch_size_out)

            # Set the outer 10% of logits_np to zero
            logits_np[:margin, :] = 0  # Top margin
            logits_np[-margin:, :] = 0  # Bottom margin
            logits_np[:, :margin] = 0  # Left margin
            logits_np[:, -margin:] = 0  # Right margin

            # Determine the location in the stitched_result array
            out_y_start = y * stride // patch_size * patch_size_out
            out_y_end = out_y_start + patch_size_out
            out_x_start = x * stride // patch_size * patch_size_out
            out_x_end = out_x_start + patch_size_out

            # Add the result to the stitched_result array and increment prediction counts
            out_arr[out_y_start:out_y_end, out_x_start:out_x_end] += logits_np
            pred_counts[out_y_start + margin:out_y_end - margin, out_x_start + margin:out_x_end - margin] += 1

    progress_bar.close()
    # Average the predictions
    out_arr /= pred_counts

    return out_arr


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python infer_full_fragment.py <checkpoint_path> <fragment_num>")

    checkpoint_path = sys.argv[1]
    fragment_num = sys.argv[2]

    # inference
    result = infer_full_fragment(fragment_num, checkpoint_path)

    date_time_string = datetime.now().strftime("%Y%m%d-%H%M%S")
    results_dir = os.path.join("inference", "results", f"fragment{fragment_num}", date_time_string)
    os.makedirs(results_dir, exist_ok=True)

    # save logits
    plt.imshow(result, cmap='gray')
    logit_path = os.path.join(results_dir, f"logits_fragment{fragment_num}_{date_time_string}.png")
    plt.savefig(logit_path, bbox_inches='tight', dpi=500, pad_inches=0)

    # save binary
    plt.imshow(result > 0.5, cmap='gray')
    binary_bath = os.path.join(results_dir, f"binarized_fragment{fragment_num}_{date_time_string}.png")
    plt.savefig(binary_bath, bbox_inches='tight', dpi=1500, pad_inches=0)

    print("Saved results to", results_dir)
    # save raw output
    np.save(os.path.join(results_dir, f"frag_{fragment_num}_{date_time_string}result.npy"), result)

    if CFG.local:
        plt.imshow(result, cmap='gray')
        plt.imshow(result > 0.5, cmap='gray')
        plt.show()
