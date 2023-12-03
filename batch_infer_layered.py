import os
import sys
from datetime import datetime

import albumentations as A
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm
from transformers import SegformerForSemanticSegmentation

from config_handler import Config
from constants import get_frag_name_from_id

'''
        SET WORKING DIRECTORY TO PROJECT ROOT

        Runs inference on full image, by slicing it into patches.
'''

val_image_aug = [
    A.Normalize(mean=[0], std=[1]),
]


def read_fragment(work_dir, fragment_id, layer_start, layer_count):
    images = []

    for i in range(layer_start, layer_start + layer_count):
        img_path = os.path.join(work_dir, "data", "fragments", f"fragment{fragment_id}", "slices", f"{i:05}.tif")

        image = cv2.imread(img_path, 0)
        assert 1 < np.asarray(image).max() <= 255, "Invalid image"

        images.append(image)
    images = np.stack(images, axis=0)

    return images


def infer_full_fragment_layer(model, fragment_id, config: Config, layer_start):
    patch_size = config.patch_size
    expected_patch_shape = (config.in_chans, patch_size, patch_size)

    # Loading images
    images = read_fragment(work_dir=config.work_dir, fragment_id=fragment_id, layer_start=layer_start,
                           layer_count=config.in_chans)

    # Hyperparams
    label_size = config.label_size
    margin_percent = 0.1
    stride_factor = 2

    margin = int(margin_percent * label_size)
    mask = np.ones((label_size, label_size), dtype=bool)
    mask[margin:-margin, margin:-margin] = False

    stride = patch_size // stride_factor
    stride_out = label_size // stride_factor

    height, width = images[0].shape

    # Calculate the number of patches needed, considering the stride
    x_patches = int(np.ceil((width - patch_size) / stride)) + 1
    y_patches = int(np.ceil((height - patch_size) / stride)) + 1

    # Initialize arrays to hold the stitched result and count of predictions for each pixel
    out_height = y_patches * stride_out + label_size
    out_width = x_patches * stride_out + label_size
    out_arr = np.zeros((out_height, out_width), dtype=np.float32)
    pred_counts = np.zeros((out_height, out_width), dtype=np.int32)

    progress_bar = tqdm(total=x_patches * y_patches, desc=f"Step {layer_start}/{end_idx}: Infer Full Fragment "
                                                          f"{get_frag_name_from_id(fragment_id)}: Processing patches"
                                                          f" for layers {layer_start}-{layer_start + config.in_chans - 1}")

    batch_size = 2  # default for 528
    if config.patch_size == 256:
        batch_size = 8

    batches = []
    batch_indices = []

    def process_patch(logits_np, x, y):
        # Calculate the margin to ignore (10% of the patch size)
        # Set the outer 10% of averaged_logits to zero
        logits_np[mask] = 0

        # Determine the location in the stitched_result array
        out_y_start = y * stride_out
        out_y_end = out_y_start + label_size
        out_x_start = x * stride_out
        out_x_end = out_x_start + label_size

        # Add the result to the stitched_result array and increment prediction counts
        out_arr[out_y_start:out_y_end, out_x_start:out_x_end] += logits_np
        pred_counts[out_y_start + margin:out_y_end - margin, out_x_start + margin:out_x_end - margin] += 1

    transform = A.Compose(val_image_aug, is_check_shapes=False)

    for y in range(y_patches):
        for x in range(x_patches):
            progress_bar.update(1)

            x_start = x * stride
            x_end = min(x_start + patch_size, width)
            y_start = y * stride
            y_end = min(y_start + patch_size, height)

            patch = images[:, y_start:y_end, x_start:x_end]

            # If the patch size is smaller than the expected size, skip it
            if patch.shape != expected_patch_shape:
                continue

            batches.append(patch)
            batch_indices.append((x, y))

            # When batch is full, process it
            if len(batches) == batch_size:
                transformed_images = [transform(image=image)['image'] for image in batches]

                batch_tensor = torch.tensor(np.stack(transformed_images)).float()
                batch_tensor = batch_tensor.to("cuda")
                outputs = model(batch_tensor)
                logits = outputs.logits
                logits_np = torch.sigmoid(logits).detach().squeeze().cpu().numpy()

                for idx, (x, y) in enumerate(batch_indices):
                    process_patch(logits_np[idx], x, y)  # Function to process each patch

                batches = []
                batch_indices = []

    # Process any remaining patches
    if batches:
        transformed_images = [transform(image=image)['image'] for image in batches]

        batch_tensor = torch.tensor(np.stack(transformed_images)).float()
        batch_tensor = batch_tensor.to("cuda")
        outputs = model(batch_tensor)
        logits = outputs.logits
        logits_np = torch.sigmoid(logits).detach().squeeze().cpu().numpy()
        if len(batches) == 1:
            logits_np = np.expand_dims(logits_np, axis=0)

        for idx, (x, y) in enumerate(batch_indices):
            process_patch(logits_np[idx], x, y)

    progress_bar.close()

    # Average the predictions
    out_arr = np.divide(out_arr, pred_counts, where=(pred_counts > 0))

    return out_arr


if __name__ == '__main__':
    if len(sys.argv) != 4:
        print("Usage: python batch_infer_layered.py <config_path> <checkpoint_path> <fragment_id>")
        sys.exit(1)

    config_path = sys.argv[1]
    checkpoint_path = sys.argv[2]
    fragment_id = sys.argv[3]

    config = Config.load_from_file(config_path)
    channels = config.in_chans

    date_time_string = datetime.now().strftime("%Y%m%d-%H%M%S")
    model_run_name = '-'.join(checkpoint_path.split(f"checkpoints{os.sep}")[-1].split('-')[0:5])
    root_dir = os.path.join("inference", "results", f"fragment{fragment_id}")
    results_dir = os.path.join("inference", "results", f"fragment{fragment_id}", f"{date_time_string}_{model_run_name}")

    os.makedirs(root_dir, exist_ok=True)

    dirs = [x for x in os.listdir(root_dir) if x.endswith(model_run_name)]
    if len(dirs) == 1:
        results_dir = os.path.join(root_dir, dirs[0])

    os.makedirs(results_dir, exist_ok=True)
    print(f"Created directory {results_dir}")

    model = SegformerForSemanticSegmentation.from_pretrained(config.from_pretrained,
                                                             num_labels=1,
                                                             num_channels=config.in_chans,
                                                             ignore_mismatched_sizes=True)
    model = model.to("cuda")
    checkpoint = torch.load(checkpoint_path)
    state_dict = {key.replace('model.', ''): value for key, value in checkpoint['state_dict'].items()}
    model.load_state_dict(state_dict)
    print("Loaded model", checkpoint_path)

    output_arrays = []

    start_idx = None
    end_idx = None

    if not start_idx:
        start_idx = 0
    if not end_idx:
        end_idx = 63

    for i in range(start_idx, end_idx, 1):
        sigmoid_logits = infer_full_fragment_layer(model=model,
                                                   fragment_id=fragment_id,
                                                   config=config,
                                                   layer_start=i)

        output_arrays.append(sigmoid_logits)
        np.save(os.path.join(results_dir, f"sigmoid_logits_{i}_{i + channels - 1}.npy"), sigmoid_logits)

    maxed_arr = np.maximum.reduce(output_arrays)
    plt.imshow(maxed_arr, cmap='gray')  # Change colormap if needed
    plt.colorbar()
    plt.savefig(os.path.join(results_dir, "maxed_logits.png"), dpi=500, )
    np.save(os.path.join(results_dir, "maxed_logits.npy"), maxed_arr)
