import argparse
import os
import sys
from datetime import datetime

import albumentations as A
import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import SegformerForSemanticSegmentation

from config_handler import Config
from constants import get_frag_name_from_id, get_ckpt_name_from_id
from models.simplecnn import SimpleCNNModule

'''
        SET WORKING DIRECTORY TO PROJECT ROOT

        Runs inference on full image, by slicing it into patches.
'''

val_image_aug = [
    A.Normalize(mean=[0], std=[1]),
]


def pad_image_to_be_divisible_by_4(image, patch_size):
    # Calculate the padding required for height and width
    height_pad = -image.shape[0] % patch_size
    width_pad = -image.shape[1] % patch_size

    # Pad the image
    padded_image = np.pad(image, ((0, height_pad), (0, width_pad)), mode='constant')

    return padded_image


def read_fragment(patch_size, work_dir, fragment_id, layer_start, layer_count):
    images = []

    for i in range(layer_start, layer_start + layer_count):
        img_path = os.path.join(work_dir, "data", "fragments", f"fragment{fragment_id}", "slices", f"{i:05}.tif")

        image = cv2.imread(img_path, 0)
        assert 1 < np.asarray(image).max() <= 255, "Invalid image"

        image = pad_image_to_be_divisible_by_4(image, patch_size)

        images.append(image)
    images = np.stack(images, axis=0)

    return images


def infer_full_fragment_layer(model, ckpt_name, batch_size, fragment_id, config: Config, layer_start):
    patch_size = config.patch_size
    expected_patch_shape = (config.in_chans, patch_size, patch_size)

    # Loading images
    images = read_fragment(patch_size=patch_size, work_dir=config.work_dir, fragment_id=fragment_id,
                           layer_start=layer_start, layer_count=config.in_chans)

    # Load mask
    mask_path = os.path.join(config.work_dir, "data", "fragments", f"fragment{fragment_id}", "mask.png")
    if not os.path.isfile(mask_path):
        raise ValueError(f"Mask file does not exist for fragment: {fragment_id}")
    mask = np.asarray(Image.open(mask_path))

    mask = pad_image_to_be_divisible_by_4(mask, patch_size)

    assert mask.shape == images[0].shape, "Mask shape does not match image shape"

    # Hyperparams
    label_size = config.label_size
    margin_percent = 0.1
    stride_factor = 2

    margin = int(margin_percent * label_size)
    ignore_edge_mask = torch.ones((label_size, label_size), dtype=torch.bool, device='cuda')
    ignore_edge_mask[margin:-margin, margin:-margin] = False

    stride = patch_size // stride_factor
    stride_out = label_size // stride_factor

    height, width = images[0].shape
    assert height % patch_size == 0 and width % patch_size == 0, "Input height/width not valid"

    # Calculate the number of patches needed, considering the stride
    x_patches = int((width / stride) - 1)
    y_patches = int((height / stride) - 1)

    # Initialize arrays to hold the stitched result and count of predictions for each pixel
    out_height = height // 4
    out_width = width // 4

    assert out_height % 128 == 0 and out_width % 128 == 0, "Out height/width not valid"

    out_arr = torch.zeros((out_height, out_width), dtype=torch.float16, device='cuda')
    pred_counts = torch.zeros((out_height, out_width), dtype=torch.int16, device='cuda')

    progress_bar = tqdm(total=x_patches * y_patches,
                        desc=f"Step {layer_start}/{end_idx - 1}: Infer Fragment {get_frag_name_from_id(fragment_id)} "
                             f"with {get_ckpt_name_from_id(ckpt_name).upper()}: Processing patches"
                             f" for layers {layer_start}-{layer_start + config.in_chans - 1}")

    preallocated_batch_tensor = torch.zeros((batch_size, *expected_patch_shape), dtype=torch.float16, device='cuda')
    model = model.half()

    batches = []
    batch_indices = []

    def process_patch(logits, x, y):
        # Calculate the margin to ignore (10% of the patch size)
        # Set the outer 10% of averaged_logits to zero
        logits *= ~ignore_edge_mask

        # Determine the location in the stitched_result array
        out_y_start = y * stride_out
        out_y_end = out_y_start + label_size
        out_x_start = x * stride_out
        out_x_end = out_x_start + label_size

        # Add the result to the stitched_result array and increment prediction counts
        out_arr[out_y_start:out_y_end, out_x_start:out_x_end] += logits
        pred_counts[out_y_start + margin:out_y_end - margin, out_x_start + margin:out_x_end - margin] += 1

    transform = A.Compose(val_image_aug, is_check_shapes=False)

    for y in range(y_patches):
        for x in range(x_patches):
            progress_bar.update(1)

            x_start = x * stride
            x_end = x_start + patch_size
            y_start = y * stride
            y_end = y_start + patch_size

            mask_patch = mask[y_start:y_end, x_start:x_end]
            # is mask is completely zero, ignore patch
            if np.all(mask_patch == 0):
                continue

            patch = images[:, y_start:y_end, x_start:x_end]

            # If the patch size is smaller than the expected size, skip it
            if patch.shape != expected_patch_shape:
                continue

            batches.append(patch)
            batch_indices.append((x, y))

            # When batch is full, process it
            if len(batches) == batch_size:
                transformed_images = [transform(image=image)['image'] for image in batches]

                for idx, patch in enumerate(transformed_images):
                    preallocated_batch_tensor[idx] = torch.from_numpy(patch).float()

                with torch.no_grad():
                    outputs = model(preallocated_batch_tensor[:len(batches)])

                logits = torch.sigmoid(outputs.logits).detach().squeeze()

                for idx, (x, y) in enumerate(batch_indices):
                    process_patch(logits[idx], x, y)  # Function to process each patch

                batches = []
                batch_indices = []

    # Process any remaining patches
    if batches:
        transformed_images = [transform(image=image)['image'] for image in batches]

        for idx, patch in enumerate(transformed_images):
            preallocated_batch_tensor[idx] = torch.from_numpy(patch).float()

        outputs = model(preallocated_batch_tensor[:len(batches)])
        logits = torch.sigmoid(outputs.logits).detach().squeeze(dim=1)

        if len(batches) == 1:
            logits.unsqueeze(dim=0)

        for idx, (x, y) in enumerate(batch_indices):
            process_patch(logits[idx], x, y)

    progress_bar.close()

    # Average the predictions
    out_arr = torch.where(pred_counts > 0, torch.div(out_arr, pred_counts), out_arr)

    return out_arr


def find_ckpt_in_dir(path):
    for file in os.listdir(path):
        if file.endswith('.ckpt'):
            return os.path.join(path, file)
    return None


def find_py_in_dir(path):
    for file in os.listdir(path):
        if file.endswith('.py'):
            return os.path.join(path, file)
    return None


def parse_args():
    parser = argparse.ArgumentParser(description='Batch Infer Layered Script')
    parser.add_argument('checkpoint_folder_name', type=str, help='Checkpoint folder name')
    parser.add_argument('fragment_id', type=str, help='Fragment ID')
    parser.add_argument('--start_idx', type=int, default=0, help='Start index (default: 0)')
    parser.add_argument('--end_idx', type=int, default=60, help='End index (default: 60)')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size (default: 16)')

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    Image.MAX_IMAGE_PIXELS = None
    args = parse_args()

    checkpoint_folder_name = args.checkpoint_folder_name
    fragment_id = args.fragment_id
    start_idx = args.start_idx
    end_idx = args.end_idx
    batch_size = args.batch_size

    # Determine the path to the configuration based on the checkpoint folder
    checkpoint_folder_path = os.path.join('checkpoints', checkpoint_folder_name)

    # Find config path
    config_path = find_py_in_dir(checkpoint_folder_path)

    # Find the checkpoint path
    checkpoint_path = find_ckpt_in_dir(checkpoint_folder_path)

    if checkpoint_path is None:
        print("No valid checkpoint file found")
        sys.exit(1)

    config = Config.load_from_file(config_path)

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

    if config.architecture == 'segformer':
        model = SegformerForSemanticSegmentation.from_pretrained(config.from_pretrained,
                                                                 num_labels=1,
                                                                 num_channels=config.in_chans,
                                                                 ignore_mismatched_sizes=True)

    elif config.architecture == 'simplecnn':
        model = SimpleCNNModule(cfg=config)
    else:
        print("Error")
        sys.exit(1)

    model = model.to("cuda")

    checkpoint = torch.load(checkpoint_path)
    state_dict = {key.replace('model.', ''): value for key, value in checkpoint['state_dict'].items()}
    model.load_state_dict(state_dict)
    print("Loaded model", checkpoint_path)

    for i in range(start_idx, end_idx + 1, 1):
        file_path = os.path.join(results_dir, f"sigmoid_logits_{i}_{i + config.in_chans - 1}.npy")
        if os.path.isfile(file_path):
            continue

        sigmoid_logits = infer_full_fragment_layer(model=model,
                                                   ckpt_name=checkpoint_folder_name,
                                                   batch_size=batch_size,
                                                   fragment_id=fragment_id,
                                                   config=config,
                                                   layer_start=i)
        torch.cuda.empty_cache()
        output = sigmoid_logits.cpu().numpy()
        np.save(file_path, output)
