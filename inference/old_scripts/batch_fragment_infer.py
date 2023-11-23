import os
import sys
from datetime import datetime

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from skimage.transform import resize
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
        img_path = os.path.join(CFG.data_root_dir, "fragments", f"fragment{fragment_id}", "slices", f"{i:05}.tif")

        image = cv2.imread(img_path, 0)
        assert 1 < np.asarray(image).max() <= 255, "Invalid image"

        pad0 = (CFG.patch_size - image.shape[0] % CFG.patch_size) % CFG.patch_size
        pad1 = (CFG.patch_size - image.shape[1] % CFG.patch_size) % CFG.patch_size

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


def infer_full_fragment(fragment_index, checkpoint_path, batch_size=8):
    images = read_fragment(fragment_index)

    # Load your model
    model = SegformerForSemanticSegmentation.from_pretrained(CFG.seg_pretrained, num_labels=1, num_channels=16,
                                                             ignore_mismatched_sizes=True)
    checkpoint = torch.load(checkpoint_path)
    state_dict = {key.replace('model.', ''): value for key, value in checkpoint['state_dict'].items()}
    model.load_state_dict(state_dict)
    print("Loaded model", checkpoint_path)
    # Define the size of the patches
    patch_size = 512
    height, width = images[0].shape

    # Calculate the number of patches needed, ensuring coverage of the entire image
    x_patches = int(np.ceil(width / patch_size))
    y_patches = int(np.ceil(height / patch_size))

    # Initialize an empty array to hold the stitched result
    stitched_result = np.zeros((y_patches * patch_size, x_patches * patch_size), dtype=np.float32)

    progress_bar = tqdm(total=x_patches * y_patches, desc="Infer Full Fragment: Processing patches")

    batch = []
    batch_data = np.zeros((batch_size, CFG.in_chans, patch_size, patch_size))  # Assuming 'channels' is defined
    batch_info = []
    batch_count = 0

    for y in range(y_patches):
        for x in range(x_patches):
            progress_bar.update(1)

            x_start = x * patch_size
            x_end = min(x_start + patch_size, width)
            y_start = y * patch_size
            y_end = min(y_start + patch_size, height)

            patch = images[:, y_start:y_end, x_start:x_end]

            # If the patch size is smaller than the expected size, pad it
            if patch.shape[1] < patch_size or patch.shape[2] < patch_size:
                padding_height = max(patch_size - patch.shape[1], 0)
                padding_width = max(patch_size - patch.shape[2], 0)
                patch = np.pad(patch, ((0, 0), (0, padding_height), (0, padding_width)), 'constant')

            batch_data[batch_count] = patch
            batch_info.append((x_start, x_end, y_start, y_end))
            batch_count += 1

            # Process the batch when it's full or at the end of the loop
            if batch_count == batch_size or (y == y_patches - 1 and x == x_patches - 1):
                batch_patches = torch.tensor(batch_data[:batch_count]).float()
                outputs = model(batch_patches)
                logits = outputs.logits.detach().cpu().numpy()

                for i, (x_start, x_end, y_start, y_end) in enumerate(batch_info):
                    logits_np = resize(logits[i].squeeze(), (patch_size, patch_size), order=0, preserve_range=True, anti_aliasing=False)
                    stitched_result[y_start:y_end, x_start:x_end] = logits_np[:y_end - y_start, :x_end - x_start]

                batch_count = 0
                batch_info = []

    progress_bar.close()

    # Crop the stitched_result to the original image size
    stitched_result = stitched_result[:height, :width]

    return stitched_result


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
    # np.save(f"frag_{fragment_num}_{date_time_string}result.npy", result)

    if CFG.local:
        plt.imshow(result, cmap='gray')
        plt.imshow(result > 0.5, cmap='gray')
        plt.show()
