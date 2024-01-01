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
from tqdm import tqdm
from transformers import SegformerForSemanticSegmentation
from transformers.utils import logging

from models.architectures.unet3d_segformer import UNET3D_Segformer
from models.architectures.unetr_segformer import UNETR_Segformer
from utility.checkpoints import get_ckpt_name_from_id
from utility.configs import Config
from utility.fragments import get_frag_name_from_id, FragmentHandler, SUPERSEDED_FRAGMENTS

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


def read_fragment(contrasted, patch_size, work_dir, fragment_id, layer_start, layer_count):
    print("Start reading images")
    images = []

    for i in tqdm(range(layer_start, layer_start + layer_count)):

        fragment = "fragments_contrasted" if contrasted else "fragments"
        img_path = os.path.join(work_dir, "data", fragment, f"fragment{fragment_id}", "slices", f"{i:05}.tif")

        if not os.path.isfile(img_path):
            if contrasted:
                print(f"Missing Contrasted Slice file: {os.path.join(fragment_id, 'slices', f'{i:05}.tif')}")
                exit()

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


def advanced_tta(model, tensor):
    """
    Apply test-time augmentation to the input tensor and make a batch inference with PyTorch.

    :param tensor: Image tensor with shape (16, 512, 512).
    :param rotate: Apply rotation if True.
    :param flip_vertical: Apply vertical flip if True.
    :param flip_horizontal: Apply horizontal flip if True.
    :return: Batch of TTA-processed tensors.
    """
    with torch.no_grad():
        tta_batch = []
        tensor = tensor.squeeze()  # 16, 512, 512
        tta_batch.append(tensor.clone())

        # Apply rotation augmentations
        for k in [1, 2, 3]:
            rotated = torch.rot90(tensor, k, [1, 2]).clone()  # 16, 512, 512
            tta_batch.append(rotated)

        # Apply flip augmentations
        vert_flipped = torch.flip(tensor, [1]).clone()
        hor_flipped = torch.flip(tensor, [2]).clone()
        z_flipped = torch.flip(tensor, [0]).clone()

        tta_batch.append(vert_flipped)  # Vertical flip
        tta_batch.append(hor_flipped)  # Horizontal flip
        tta_batch.append(z_flipped)

        for k in [1, 2, 3]:
            rotated = torch.rot90(z_flipped, k, [1, 2]).clone()  # 16, 512, 512
            tta_batch.append(rotated)

        # Convert list to torch tensor
        tta_batch = torch.stack(tta_batch).half()  # [6, 16, 512, 512]
        tta_batch = tta_batch.unsqueeze(1)  # [6, 1, 16, 512, 512]
        # print("Batch Shape before model forward:", tta_batch.shape)

        # Get the model's predictions for the batch
        tta_outputs = model(tta_batch)  # (6, 128, 128)
        # print("Batch Shape after model forward:", tta_outputs.shape)

        # Post-process to revert the TTA
        reverted_outputs = []
        for i, output in enumerate(tta_outputs):
            output = output.clone()
            if i == 1:  # Revert rotate left
                output = torch.rot90(output, 3, [0, 1])
            elif i == 2:  # Revert rotate 180
                output = torch.rot90(output, 2, [0, 1])
            elif i == 3:  # Revert rotate right
                output = torch.rot90(output, 1, [0, 1])
            elif i == 4:  # Revert vertical flip
                output = torch.flip(output, [0])
            elif i == 5:  # Revert horizontal flip
                output = torch.flip(output, [1])
            elif i == 6:  # Z-flip, nothing to revert
                pass
            elif i == 7:  # Revert rotate left (of z flipped)
                output = torch.rot90(output, 3, [0, 1])
            elif i == 8:  # Revert rotate 180 (of z flipped)
                output = torch.rot90(output, 2, [0, 1])
            elif i == 9:  # Revert rotate right (of z flipped)
                output = torch.rot90(output, 1, [0, 1])
            reverted_outputs.append(output.clone().squeeze())

        stacked_outputs = torch.stack(reverted_outputs)
        sigmoided_outputs = torch.sigmoid(stacked_outputs).detach()

        return sigmoided_outputs.mean(dim=0)


def infer_full_fragment_layer(model, npy_file_path, ckpt_name, stride_factor, fragment_id, config: Config,
                              layer_start, gpu, resume_arr):
    resuming = resume_arr is not None
    print("Starting full inference")
    patch_size = config.patch_size
    expected_patch_shape_padded = (1, config.in_chans + 4, patch_size, patch_size)
    expected_patch_shape_extracted = (12, patch_size, patch_size)

    contrasted = getattr(config, 'contrasted', False)

    # Loading images [12, Height, Width]
    images = read_fragment(contrasted=contrasted, patch_size=patch_size, work_dir=config.work_dir,
                           fragment_id=fragment_id, layer_start=layer_start, layer_count=config.in_chans)

    # Load mask
    mask_path = os.path.join(config.work_dir, "data", "fragments", f"fragment{fragment_id}", "mask.png")
    if not os.path.isfile(mask_path):
        raise ValueError(f"Mask file does not exist for fragment: {fragment_id}")
    mask = np.asarray(Image.open(mask_path))
    mask = pad_image_to_be_divisible_by_4(mask, patch_size)

    assert mask.shape == images[0].shape, f"Mask shape {mask.shape} does not match image shape {images[0].shape}"

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
    if resuming:
        resume_tensor = torch.tensor(resume_arr, dtype=torch.float16, device=f'cuda:{gpu}')
        out_arr[:] = resume_tensor

    pred_counts = torch.zeros((out_height, out_width), dtype=torch.int16, device=f'cuda:{gpu}')

    progress_bar = tqdm(total=x_patches * y_patches,
                        desc=f"Layer {layer_start}: Infer Fragment {get_frag_name_from_id(fragment_id)} "
                             f"with {get_ckpt_name_from_id(ckpt_name).upper()}: Processing patches"
                             f" for layers {layer_start}-{layer_start + config.in_chans - 1}")

    model = model.half()

    def process_patch(logits, x, y, out_y_start, out_y_end, out_x_start, out_x_end):
        # Calculate the margin to ignore (10% of the patch size)
        # Set the outer 10% of averaged_logits to zero
        logits = logits.clone()
        logits *= ~ignore_edge_mask

        # Add the result to the stitched_result array and increment prediction counts
        out_arr[out_y_start:out_y_end, out_x_start:out_x_end] += logits
        pred_counts[out_y_start + margin:out_y_end - margin, out_x_start + margin:out_x_end - margin] += 1

    transform = A.Compose(val_image_aug, is_check_shapes=False)

    patch_counter = 0
    for y in range(y_patches):
        for x in range(x_patches):
            progress_bar.update(1)

            # Determine the location in the stitched_result array
            out_y_start = y * stride_out
            out_y_end = out_y_start + label_size
            out_x_start = x * stride_out
            out_x_end = out_x_start + label_size

            x_start = x * stride
            x_end = x_start + patch_size
            y_start = y * stride
            y_end = y_start + patch_size

            mask_patch = mask[y_start:y_end, x_start:x_end]
            # is mask is completely zero, ignore patch
            if np.all(mask_patch == 0):
                continue

            if resuming:
                # don't process patch if there is already a result for it (if resuming)
                if torch.any(out_arr[out_y_start:out_y_end, out_x_start:out_x_end] != 0):
                    continue

            patch = images[:, y_start:y_end, x_start:x_end]  # [12, 512, 512]

            if patch.shape != expected_patch_shape_extracted:
                # patch is at the edge => skip it
                continue

            patch = np.expand_dims(patch, 0)  # [1, 12, 512, 512]
            zero_padding = np.zeros((1, 16 - patch.shape[1], patch_size, patch_size))  # [1, 4, 512, 512]
            patch = np.concatenate([patch, zero_padding], axis=1)  # [1, 16, 512, 512]

            # If the patch size is smaller than the expected size, skip it
            if patch.shape != expected_patch_shape_padded:
                continue

            # apply necessary transformations
            patch = transform(image=patch)['image']
            patch_tensor = torch.from_numpy(patch).float().to(f'cuda:{gpu}')

            sigmoid_tta_output = advanced_tta(model=model, tensor=patch_tensor)

            process_patch(sigmoid_tta_output, x, y,
                          out_y_start=out_y_start, out_y_end=out_y_end,
                          out_x_start=out_x_start, out_x_end=out_x_end)

            patch_counter += 1
            if patch_counter % 100 == 0:
                print("Saving")
                np.save(npy_file_path, out_arr.cpu().numpy())

    progress_bar.close()

    # Average the predictions
    out_arr = torch.where(pred_counts > 0, torch.div(out_arr, pred_counts), out_arr)

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

            if not os.path.isfile(img_path):
                print(f"Downloading Slice file for dimensions: {os.path.join(frag_id, 'slices', f'{i:05}.tif')}")
                command = ['bash', "./scripts/utils/download_fragment.sh", frag_id, f'{i:05} {i:05}']
                subprocess.run(command, check=True)

            image = cv2.imread(img_path, 0)
            target_dims = image.shape

    return target_dims


def normalize_npy_preds(array):
    min_value = array.min()
    max_value = array.max()
    normalized_array = (array - min_value) / (max_value - min_value)

    return normalized_array


def parse_args():
    warnings.filterwarnings('ignore', category=UserWarning, module='albumentations.*')
    logging.set_verbosity_error()
    Image.MAX_IMAGE_PIXELS = None

    parser = argparse.ArgumentParser(description='Infer Layered Script')
    parser.add_argument('checkpoint_folder_name', type=str, help='Checkpoint folder name')
    parser.add_argument('fragment_id', type=str, help='Fragment ID')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size (default: 16)')
    parser.add_argument('--resume', action='store_true', help='Continue a previously stopped inference')
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

    model = model.to(f"cuda:{gpu}")

    return model, full_model_path


def get_inference_range(frag_id):
    start_best_layer_idx, end_best_layer_idx = FragmentHandler().get_best_layers(frag_id=frag_id)

    assert start_best_layer_idx is not None and end_best_layer_idx is not None, f"No best layers found for {frag_id}"

    start_best, end_best = FragmentHandler().get_best_12_layers(frag_id=frag_id)
    if start_best and end_best:
        start_best_layer_idx = min(start_best_layer_idx, start_best)
        end_best_layer_idx = max(end_best_layer_idx, end_best)

    assert end_best_layer_idx - start_best_layer_idx >= 11, f"Not enough best layers found for 12 layer inference {frag_id}"

    return start_best_layer_idx, end_best_layer_idx


def infer_layered_with_tta(checkpoint, frag_id, stride=2, gpu=0):
    main(checkpoint, frag_id, stride, gpu)


def main(checkpoint, fragment_id, stride_factor=2, gpu=0):
    config_path = find_py_in_dir(os.path.join('checkpoints', checkpoint))
    config = Config.load_from_file(config_path)

    model, model_path = load_model(cfg=config, model_path=checkpoint, gpu=gpu)

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

    # -1 gives user the choice
    # choice = -1
    # 3 hardcodes resuming
    choice = 1
    for start_idx in range(start_best_layer_idx, end_best_layer_idx - (config.in_chans - 1) + 1):
        end_idx = start_idx + (config.in_chans - 1)

        # Check if this is the last possible N-layer range within the given range
        if end_idx > end_best_layer_idx:
            break

        npy_file_path = os.path.join(results_dir,
                                     f"tta-stride-{stride_factor}-sigmoid_logits_{start_idx}_{end_idx}.npy")

        if choice == -1 and os.path.isfile(npy_file_path):
            print(f"Inference file for {start_idx} already exists. Choose action for this and following files:")
            print("1: Skip existing files")
            print("2: Overwrite files from scratch")
            print("3: Resume existing files if they are incomplete (skips them if they are full)")
            valid_choices = [1, 2, 3]
            while choice not in valid_choices:
                choice = int(input())
                if choice in valid_choices:
                    break
                print("Invalid choice, choose one of [1] Skip [2] Overwrite [3] Resume")

        # skip if existing
        if choice == 1 and os.path.isfile(npy_file_path):
            print("Found existing, skipping ", npy_file_path)
            continue

        resume_arr = None
        # Check if prediction NPY file already exists
        if os.path.isfile(npy_file_path) and choice == 3:
            print(f"Found partial result: {npy_file_path}, resuming inference ...")
            resume_arr = np.load(npy_file_path)

        # Process each N-layer range
        infer_full_fragment_layer(model=model,
                                  ckpt_name=checkpoint,
                                  stride_factor=stride_factor,
                                  fragment_id=fragment_id,
                                  config=config,
                                  gpu=gpu,
                                  layer_start=start_idx,
                                  npy_file_path=npy_file_path,
                                  resume_arr=resume_arr)


if __name__ == '__main__':
    args = parse_args()

    main(checkpoint=args.checkpoint_folder_name,
         fragment_id=args.fragment_id,
         stride_factor=args.stride,
         gpu=args.gpu)
