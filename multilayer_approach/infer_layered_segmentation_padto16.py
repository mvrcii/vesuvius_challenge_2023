import argparse
import os
import subprocess
import sys
import warnings
from datetime import datetime

import albumentations as A
import cv2
import numpy as np
import torch
from PIL import Image
from PIL.Image import Resampling
from tqdm import tqdm
from transformers import SegformerForSemanticSegmentation
from transformers.utils import logging

from models.architectures.unet3d_segformer import UNET3D_Segformer
from models.architectures.unetr_segformer import UNETR_Segformer
from utility.checkpoints import get_ckpt_name_from_id
from utility.configs import Config
from utility.fragments import get_frag_name_from_id, FragmentHandler, SUPERSEDED_FRAGMENTS
from utility.meta_data import AlphaBetaMeta

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
    print("Start reading images")
    images = []

    for i in tqdm(range(layer_start, layer_start + layer_count)):
        img_path = os.path.join(work_dir, "data", "fragments", f"fragment{fragment_id}", "slices", f"{i:05}.tif")

        if not os.path.isfile(img_path):
            print(f"Downloading missing Slice file: {os.path.join(fragment_id, 'slices', f'{i:05}.tif')}")
            if fragment_id in SUPERSEDED_FRAGMENTS:
                print("Warning: Fragment superseded, added suffix for download!")
                fragment_id += "_superseded"
            command = ['bash', "./scripts/utils/download_fragment.sh", fragment_id, f'{i:05} {i:05}']
            subprocess.run(command, check=True)

        image = cv2.imread(img_path, 0)
        assert 1 < np.asarray(image).max() <= 255, "Invalid image index {}".format(i)

        image = pad_image_to_be_divisible_by_4(image, patch_size)

        images.append(image)
    images = np.stack(images, axis=0)

    return images


def advanced_tta(model, tensor, rotate=False, flip_vertical=False, flip_horizontal=False):
    """
    Apply test-time augmentation to the input tensor and make a batch inference with PyTorch.

    :param tensor: Image tensor with shape (4, 512, 512).
    :param rotate: Apply rotation if True.
    :param flip_vertical: Apply vertical flip if True.
    :param flip_horizontal: Apply horizontal flip if True.
    :return: Batch of TTA-processed tensors.
    """
    tta_batch = []

    # Apply rotation augmentations
    if rotate:
        tta_batch.append(torch.rot90(tensor, 1, [1, 2]).clone())  # Rotate left 90 degrees
        tta_batch.append(torch.rot90(tensor, 2, [1, 2]).clone())  # Rotate 180 degrees
        tta_batch.append(torch.rot90(tensor, 3, [1, 2]).clone())  # Rotate right 90 degrees

    # Apply flip augmentations
    if flip_vertical:
        tta_batch.append(torch.flip(tensor, [1]).clone())  # Vertical flip
    if flip_horizontal:
        tta_batch.append(torch.flip(tensor, [2]).clone())  # Horizontal flip

    # Convert list to torch tensor
    tta_batch = torch.stack(tta_batch).half()  # Assuming the model is in half precision

    # Get the model's predictions for the batch
    tta_outputs = model(tta_batch).logits

    # Post-process to revert the TTA
    reverted_outputs = []
    for i, output in enumerate(tta_outputs):
        output = output.clone()
        if rotate:
            if i == 1:  # Revert rotate left
                output = torch.rot90(output, 3, [1, 2])
            elif i == 2:  # Revert rotate 180
                output = torch.rot90(output, 2, [1, 2])
            elif i == 3:  # Revert rotate right
                output = torch.rot90(output, 1, [1, 2])
        if flip_vertical and i == len(tta_outputs) - 2 + int(flip_horizontal):
            output = torch.flip(output, [1])
        if flip_horizontal and i == len(tta_outputs) - 1:
            output = torch.flip(output, [2])
        reverted_outputs.append(output.clone().squeeze())

    return torch.stack(reverted_outputs)


def infer_full_fragment_layer(model, npy_file_path, ckpt_name, batch_size, stride_factor, fragment_id, config: Config,
                              layer_start, gpu):
    print("Starting full inference")
    patch_size = config.patch_size
    expected_patch_shape = (1, config.in_chans + 4, patch_size, patch_size)

    # Loading images [12, Height, Width]
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
    margin_percent = 0.2

    margin = int(margin_percent * label_size)
    ignore_edge_mask = torch.ones((label_size, label_size), dtype=torch.bool, device=f'cuda:{gpu}')
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

    out_arr = torch.zeros((out_height, out_width), dtype=torch.float16, device=f'cuda:{gpu}')
    pred_counts = torch.zeros((out_height, out_width), dtype=torch.int16, device=f'cuda:{gpu}')

    progress_bar = tqdm(total=x_patches * y_patches,
                        desc=f"Layer {layer_start}: Infer Fragment {get_frag_name_from_id(fragment_id)} "
                             f"with {get_ckpt_name_from_id(ckpt_name).upper()}: Processing patches"
                             f" for layers {layer_start}-{layer_start + config.in_chans - 1}")

    preallocated_batch_tensor = torch.zeros((batch_size, *expected_patch_shape), dtype=torch.float16, device=f'cuda:{gpu}')
    model = model.half()

    batches = []
    batch_indices = []

    def process_patch(logits, x, y):
        # Calculate the margin to ignore (10% of the patch size)
        # Set the outer 10% of averaged_logits to zero
        logits = logits.clone()
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
    use_advanced_tta = False

    batch_counter = 0
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

            patch = images[:, y_start:y_end, x_start:x_end]  # [12, 512, 512]

            if patch.shape != (12, 512, 512):
                # patch is at the edge => skip it
                continue

            patch = np.expand_dims(patch, 0)  # [1, 12, 512, 512]
            zero_padding = np.zeros((1, 16 - patch.shape[1], patch_size, patch_size))  # [1, 4, 512, 512]
            patch = np.concatenate([patch, zero_padding], axis=1)  # [1, 16, 512, 512]

            # If the patch size is smaller than the expected size, skip it
            if patch.shape != expected_patch_shape:
                continue

            batches.append(patch)
            batch_indices.append((x, y))

            # When batch is full, process it
            if len(batches) == batch_size:
                batch_counter += 1
                transformed_images = [transform(image=image)['image'] for image in batches]

                for idx, patch in enumerate(transformed_images):
                    print(f"Move batch to cuda:{gpu}")
                    preallocated_batch_tensor[idx] = torch.from_numpy(patch).float().to(f'cuda:{gpu}')

                with torch.no_grad():
                    outputs = model(preallocated_batch_tensor[:len(batches)])

                sigmoid_output = torch.sigmoid(outputs).detach()

                for idx, (x, y) in enumerate(batch_indices):
                    process_patch(sigmoid_output[idx], x, y)  # Function to process each patch

                batches = []
                batch_indices = []

            if batch_counter % 1000 == 0:
                # print("Saving")
                np.save(npy_file_path, out_arr.cpu().numpy())

    # Process any remaining patches
    if batches:
        transformed_images = [transform(image=image)['image'] for image in batches]

        for idx, patch in enumerate(transformed_images):
            preallocated_batch_tensor[idx] = torch.from_numpy(patch).float().to(f'cuda:{gpu}')

        with torch.no_grad():
            outputs = model(preallocated_batch_tensor[:len(batches)])
        probs = torch.sigmoid(outputs).detach()

        for idx, (x, y) in enumerate(batch_indices):
            process_patch(probs[idx], x, y)

    progress_bar.close()

    # Average the predictions
    out_arr = torch.where(pred_counts > 0, torch.div(out_arr, pred_counts), out_arr)

    if use_advanced_tta:
        global total_advanced_tta_patches
        print(f"Advanced TTA Patches for layer start {layer_start}: {total_advanced_tta_patches}")

    torch.cuda.empty_cache()
    output = out_arr.cpu().numpy()
    print("Saving finally")
    np.save(npy_file_path, output)

def find_py_in_dir(path):
    for file in os.listdir(path):
        if file.endswith('.py'):
            return os.path.join(path, file)
    return None


def get_target_dims(work_dir, frag_id):
    frag_dir = os.path.join(work_dir, "data", "fragments", f"fragment{frag_id}")
    assert os.path.isdir(frag_dir)

    target_dims = None

    slice_dir = os.path.join(frag_dir, "slices")
    if os.path.isdir(slice_dir):
        for i in range(0, 63):
            if target_dims:
                return target_dims

            img_path = os.path.join(slice_dir, f"{i:05}.tif")
            if os.path.isfile(img_path):
                image = cv2.imread(img_path, 0)
                target_dims = image.shape

    return target_dims


def normalize_npy_preds(array):
    min_value = array.min()
    max_value = array.max()
    normalized_array = (array - min_value) / (max_value - min_value)

    return normalized_array


def generate_and_save_label_file(cfg: Config, _model_name, array, frag_id, layer_index):
    binarized_label_dir = AlphaBetaMeta().get_label_target_dir()
    target_dir = os.path.join(binarized_label_dir, _model_name, frag_id)

    label_filename = f"{frag_id}_inklabels_{layer_index}_{layer_index + cfg.in_chans - 1}.png"
    label_path = os.path.join(target_dir, label_filename)

    # Check if label PNG file exists -> skip
    if os.path.isfile(label_path):
        return

    os.makedirs(target_dir, exist_ok=True)

    target_dims = get_target_dims(work_dir=cfg.work_dir, frag_id=frag_id)

    image = process_image(array=array, frag_id=frag_id, dimensions=target_dims)
    image.save(label_path)

    print("Saved label file to:", label_path)


def process_image(array, frag_id, dimensions):
    processed = normalize_npy_preds(array)  # Normalize

    threshold = 0.5

    # Binarize
    processed = np.where(processed > threshold, 1, 0)

    image = Image.fromarray(np.uint8(processed * 255), 'L')

    new_width = image.width * 4
    new_height = image.height * 4

    original_height, original_width = dimensions
    upscaled_image = image.resize((new_width, new_height), Resampling.LANCZOS)

    assert new_width >= original_width and new_height >= original_height

    width_diff = new_width - original_width
    height_diff = new_height - original_height

    out_width = -width_diff
    out_height = -height_diff

    if width_diff == 0:
        out_width = new_width

    if height_diff == 0:
        out_height = new_height

    return Image.fromarray(np.array(upscaled_image)[:out_height, :out_width])


def parse_args():
    warnings.filterwarnings('ignore', category=UserWarning, module='albumentations.*')
    logging.set_verbosity_error()
    Image.MAX_IMAGE_PIXELS = None

    parser = argparse.ArgumentParser(description='Infer Layered Script')
    parser.add_argument('checkpoint_folder_name', type=str, help='Checkpoint folder name')
    parser.add_argument('fragment_id', type=str, help='Fragment ID')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size (default: 16)')
    parser.add_argument('--labels', action='store_true', help='Additionally store labels pngs '
                                                              'for the inference')
    parser.add_argument('--stride', type=int, default=2, help='Stride (default: 2)')
    parser.add_argument('--gpu', type=int, default=0, help='Cuda GPU (default: 0)')
    args = parser.parse_args()

    return args


def load_model(cfg: Config, model_path, gpu):
    model_path = os.path.join('checkpoints', model_path)
    full_model_path = None
    for file in os.listdir(model_path):
        if file.endswith('.ckpt'):
            full_model_path = os.path.join(model_path, file)

    if full_model_path is None:
        print("No valid model checkpoint file found")
        sys.exit(1)

    if cfg.architecture == 'segformer':
        model = SegformerForSemanticSegmentation.from_pretrained(cfg.from_pretrained,
                                                                 num_labels=1,
                                                                 num_channels=cfg.in_chans,
                                                                 ignore_mismatched_sizes=True)
    elif cfg.architecture == 'unet3d-sf':
        model = UNET3D_Segformer(cfg=cfg)
    elif cfg.architecture == 'unetr-sf':
        model = UNETR_Segformer(cfg=cfg)
    else:
        print("Error model type not found:", cfg.architecture)
        sys.exit(1)

    checkpoint = torch.load(full_model_path)
    state_dict = {key.replace('model.', ''): value for key, value in checkpoint['state_dict'].items()}
    model.load_state_dict(state_dict)

    print(f"Move model to cuda:{gpu}")
    model = model.to(f"cuda:{gpu}")

    return model, full_model_path


def get_inference_range(frag_id):
    start_best_layer_idx, end_best_layer_idx = FragmentHandler().get_best_layers(frag_id=frag_id)
    assert start_best_layer_idx is not None and end_best_layer_idx is not None, f"No best layers found for {frag_id}"
    assert end_best_layer_idx - start_best_layer_idx >= 11, f"Not enough best layers found for 12 layer inference {frag_id}"
    return start_best_layer_idx, end_best_layer_idx


def main():
    args = parse_args()

    model_folder_name = args.checkpoint_folder_name
    fragment_id = args.fragment_id
    batch_size = args.batch_size
    stride_factor = args.stride
    gpu = args.gpu

    config_path = find_py_in_dir(os.path.join('checkpoints', model_folder_name))
    config = Config.load_from_file(config_path)

    model, model_path = load_model(cfg=config, model_path=model_folder_name, gpu=gpu)

    date_time_string = datetime.now().strftime("%Y%m%d-%H%M%S")
    model_name = model_path.split(f"checkpoints{os.sep}")[-1]
    model_name_modified = '-'.join(model_name.split('-')[0:5])
    root_dir = os.path.join("inference", "results", f"fragment{fragment_id}")
    results_dir = os.path.join("inference", "results", f"fragment{fragment_id}",
                               f"{date_time_string}_{model_name_modified}")

    os.makedirs(root_dir, exist_ok=True)

    # If inference dir already exists, use that directory
    dirs = [x for x in os.listdir(root_dir) if x.endswith(model_name_modified)]
    if len(dirs) == 1:
        results_dir = os.path.join(root_dir, dirs[0])

    os.makedirs(results_dir, exist_ok=True)

    start_best_layer_idx, end_best_layer_idx = get_inference_range(frag_id=fragment_id)

    for start_idx in range(start_best_layer_idx, end_best_layer_idx - (config.in_chans - 1) + 1):
        end_idx = start_idx + (config.in_chans - 1)

        # Check if this is the last possible N-layer range within the given range
        if end_idx > end_best_layer_idx:
            break

        npy_file_path = os.path.join(results_dir, f"stride-{stride_factor}-sigmoid_logits_{start_idx}_{end_idx}.npy")

        # Check if prediction NPY file already exists
        if os.path.isfile(npy_file_path):
            print(f"Inference already exists. Skip layer {start_idx}")
            continue

        # Process each N-layer range
        infer_full_fragment_layer(model=model,
                                  ckpt_name=model_folder_name,
                                  batch_size=batch_size,
                                  stride_factor=stride_factor,
                                  fragment_id=fragment_id,
                                  config=config,
                                  gpu=gpu,
                                  layer_start=start_idx,
                                  npy_file_path=npy_file_path)



if __name__ == '__main__':
    main()
