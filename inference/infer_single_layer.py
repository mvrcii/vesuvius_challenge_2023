import argparse
import os
import sys
import warnings
from datetime import datetime

import albumentations as A
import cv2
import numpy as np
import torch
from PIL import Image
from skimage.transform import resize
from tqdm import tqdm
from transformers import SegformerForSemanticSegmentation
from transformers.utils import logging

from utility.checkpoints import get_ckpt_name_from_id
from utility.configs import Config
from utility.fragments import FragmentHandler, get_frag_name_from_id

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


def read_fragment(patch_size, work_dir, fragment_id, layer_start):
    img_path = os.path.join(work_dir, "data", "fragments", f"fragment{fragment_id}", "slices",
                            f"{layer_start:05}.tif")

    if not os.path.isfile(img_path):
        print(img_path)
        print(f"{FragmentHandler().get_name(fragment_id)}: Layer File {layer_start:05}.tif not found")
        sys.exit(1)

    image = cv2.imread(img_path, 0)

    if image is None:
        print(f"{FragmentHandler().get_name(fragment_id)}: Layer File {layer_start:05}.tif could not be read successfully")
        sys.exit(1)

    assert 1 < np.asarray(image).max() <= 255, "Invalid image index {}".format(layer_start)

    image = pad_image_to_be_divisible_by_4(image, patch_size)

    return np.expand_dims(image, 0)


def infer_full_fragment_layer(model, ckpt_name, batch_size, fragment_id, config: Config, layer_start):
    patch_size = config.patch_size
    expected_patch_shape = (config.in_chans, patch_size, patch_size)

    # Loading images
    images = read_fragment(patch_size=patch_size, work_dir=config.work_dir, fragment_id=fragment_id,
                           layer_start=layer_start)

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

    out_arr = torch.zeros((out_height, out_width), dtype=torch.float16, device='cuda')
    pred_counts = torch.zeros((out_height, out_width), dtype=torch.int16, device='cuda')

    progress_bar = tqdm(total=x_patches * y_patches,
                        desc=f"Step {layer_start}: Infer Fragment {get_frag_name_from_id(fragment_id)} "
                             f"with {get_ckpt_name_from_id(ckpt_name).upper()}: Processing patches"
                             f" for layer {layer_start}")

    preallocated_batch_tensor = torch.zeros((batch_size, *expected_patch_shape), dtype=torch.float16, device='cuda')
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

                sigmoid_output = torch.sigmoid(outputs.logits).detach().squeeze()

                for idx, (x, y) in enumerate(batch_indices):
                    process_patch(sigmoid_output[idx], x, y)  # Function to process each patch

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
                if image is not None:
                    target_dims = image.shape

    return target_dims


def normalize_npy_preds(array):
    min_value = array.min()
    max_value = array.max()
    normalized_array = (array - min_value) / (max_value - min_value)

    return normalized_array


def generate_and_save_label_file(cfg: Config, _model_name, array, frag_id, layer_index):
    binarized_label_dir = os.path.join("data", "base_label_binarized_single")
    target_dir = os.path.join(binarized_label_dir, _model_name, frag_id)

    label_path = os.path.join(target_dir, f"inklabels_{layer_index}.png")

    # Check if label PNG file exists -> skip
    if os.path.isfile(label_path):
        return

    os.makedirs(target_dir, exist_ok=True)

    target_dims = get_target_dims(work_dir=cfg.work_dir, frag_id=frag_id)

    image_arr = process_image(array=array, frag_id=frag_id, dimensions=target_dims)
    cv2.imwrite(filename=label_path, img=image_arr)

    if verbose:
        print("Saved label file to:", label_path)


def process_image(array, frag_id, dimensions):
    processed = normalize_npy_preds(array)  # Normalize

    threshold = 0.5

    global boost_threshold
    if boost_threshold:
        threshold = FragmentHandler().get_boost_threshold(frag_id=frag_id)

    # Binarize
    processed = np.where(processed > threshold, 1, 0)

    image = np.uint8(processed * 255)

    new_width = image.shape[1] * 4  # width
    new_height = image.shape[0] * 4  # height

    original_height, original_width = dimensions
    upscaled_image = resize(image, (new_height, new_width),
                            order=0, preserve_range=True, anti_aliasing=False)
    assert(len(np.unique(upscaled_image)) < 3)
    assert new_width >= original_width and new_height >= original_height

    width_diff = new_width - original_width
    height_diff = new_height - original_height

    out_width = -width_diff
    out_height = -height_diff

    if width_diff == 0:
        out_width = new_width

    if height_diff == 0:
        out_height = new_height

    return np.array(upscaled_image)[:out_height, :out_width]


def parse_args():
    parser = argparse.ArgumentParser(description='Infer Layered Script')
    parser.add_argument('checkpoint_folder_name', type=str, help='Checkpoint folder name')
    parser.add_argument('fragment_id', type=str, help='Fragment ID')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size (default: 16)')
    parser.add_argument('--labels', action='store_true', help='Additionally store labels pngs '
                                                              'for the inference')
    parser.add_argument('--boost_threshold', action='store_true', help='Use a boosted threshold for saved images')
    parser.add_argument('--v', action='store_false', help='Print stuff (default True)')
    args = parser.parse_args()

    return args


def load_model(cfg: Config, model_path):
    if cfg.architecture == 'segformer':
        model = SegformerForSemanticSegmentation.from_pretrained(cfg.from_pretrained,
                                                                 num_labels=1,
                                                                 num_channels=cfg.in_chans,
                                                                 ignore_mismatched_sizes=True)
    else:
        print("Error")
        sys.exit(1)

    model = model.to("cuda")
    checkpoint = torch.load(model_path)
    state_dict = {key.replace('model.', ''): value for key, value in checkpoint['state_dict'].items()}
    model.load_state_dict(state_dict)

    return model


verbose = None
boost_threshold = None


def main():
    warnings.filterwarnings('ignore', category=UserWarning, module='albumentations.*')
    logging.set_verbosity_error()
    Image.MAX_IMAGE_PIXELS = None

    args = parse_args()

    model_folder_name = args.checkpoint_folder_name
    fragment_id = args.fragment_id
    batch_size = args.batch_size
    save_labels = args.labels

    start_layer_idx, end_layer_idx = FragmentHandler().get_best_12_layers(frag_id=fragment_id)

    global verbose, boost_threshold
    boost_threshold = args.boost_threshold
    verbose = args.v

    # Determine the path to the configuration based on the checkpoint folder
    model_folder_path = os.path.join('checkpoints', model_folder_name)

    # Find config path
    config_path = find_py_in_dir(model_folder_path)

    # Find the checkpoint path
    model_path = find_ckpt_in_dir(model_folder_path)

    if model_path is None:
        print("No valid model checkpoint file found")
        sys.exit(1)

    config = Config.load_from_file(config_path)

    date_time_string = datetime.now().strftime("%Y%m%d-%H%M%S")
    model_name = model_path.split(f"checkpoints{os.sep}")[-1]
    model_name_modified = '-'.join(model_name.split('-')[0:5])
    root_dir = os.path.join("inference", "single_results", f"fragment{fragment_id}")
    results_dir = os.path.join(root_dir, f"{date_time_string}_{model_name_modified}")

    os.makedirs(root_dir, exist_ok=True)

    dirs = [x for x in os.listdir(root_dir) if x.endswith(model_name_modified)]
    if len(dirs) == 1:
        results_dir = os.path.join(root_dir, dirs[0])

    os.makedirs(results_dir, exist_ok=True)

    model = load_model(cfg=config, model_path=model_path)

    # Calculate valid label indices
    valid_start_idxs = list(range(start_layer_idx, end_layer_idx + 1))

    for layer_idx in valid_start_idxs:
        npy_file_path = os.path.join(results_dir, f"sigmoid_logits_{layer_idx}.npy")

        # Check if prediction NPY file already exists -> skip infer
        if os.path.isfile(npy_file_path):
            npy_file = np.load(npy_file_path)

            if save_labels:
                generate_and_save_label_file(config,
                                             _model_name=model_folder_name,
                                             array=npy_file,
                                             frag_id=fragment_id,
                                             layer_index=layer_idx)
            if verbose:
                print(f"Skip inference for layer {layer_idx}")
                continue

        sigmoid_logits = infer_full_fragment_layer(model=model,
                                                   ckpt_name=model_folder_name,
                                                   batch_size=batch_size,
                                                   fragment_id=fragment_id,
                                                   config=config,
                                                   layer_start=layer_idx)

        torch.cuda.empty_cache()
        output = sigmoid_logits.cpu().numpy()
        np.save(npy_file_path, output)

        if save_labels and layer_idx in valid_start_idxs:
            generate_and_save_label_file(config,
                                         _model_name=model_folder_name,
                                         array=output,
                                         frag_id=fragment_id,
                                         layer_index=layer_idx)


if __name__ == '__main__':
    main()
