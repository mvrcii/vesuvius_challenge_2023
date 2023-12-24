import os
import re

from utility.fragments import FragmentHandler


def validate_fragments(config, fragments, label_dir):
    frag_id_2_channels = {}

    for frag_id in fragments:
        val_errors, frag_channels = validate_fragment_files(frag_id=frag_id, cfg=config,
                                                            label_dir=label_dir)
        if len(val_errors) > 0:
            print("Excluded fragment", frag_id)
            print("\n".join(val_errors))
        elif len(frag_channels) > 0:
            print("Fragment", frag_id, "is valid")
            frag_id_2_channels[frag_id] = frag_channels

    return frag_id_2_channels

def validate_fragment_files(frag_id, cfg, label_dir):
    errors = []
    frag_dir = os.path.join(cfg.work_dir, "data", "fragments", f"fragment{frag_id}")
    frag_label_dir = os.path.join(label_dir, f"{frag_id}")
    frag_slice_dir = os.path.join(frag_dir, 'slices')
    inklabel_path = os.path.join(frag_label_dir, f"{frag_id}_inklabels.png")
    ignore_path = os.path.join(frag_label_dir, f"{frag_id}_ignore.png")
    mask_path = os.path.join(frag_dir, "mask.png")

    # check if fragment directory exists (data/fragments/fragment{frag_id})
    if not os.path.isdir(frag_dir):
        errors.append(f"\033[91mReason:\t\tFragment directory '{frag_dir}' does not exist\033[0m")

    # Check if label dir (multilayer_approach/base_labels/3_binarized/fragment{frag_id}) exists
    if not os.path.isdir(frag_label_dir):
        errors.append(f"\033[91mReason:\t\tLabel directory {frag_label_dir} not found\033[0m")

    # Check if slice dir (data/fragments/fragment{frag_id}/slices) exists
    if not os.path.isdir(frag_slice_dir):
        errors.append(f"\033[91mReason:\t\tSlice directory {frag_slice_dir} not found\033[0m")

    # Check if inklabel exists
    if not os.path.isfile(inklabel_path):
        errors.append(f"\033[91mReason:\t\tInklabel file {inklabel_path} not found\033[0m")

    # Check if ignore exists
    if not os.path.isfile(ignore_path):
        errors.append(f"\033[91mReason:\t\tIgnore file {ignore_path} not found\033[0m")

    # Check if mask exists
    if not os.path.isfile(mask_path):
        errors.append(f"\033[91mReason:\t\tMask file not found\033[0m")

    # Stop if any errors occurred
    if len(errors) > 0:
        print("errors occured")
        print(errors)
        return errors, []

    # Get required 12 channels for this fragment
    required_channels_start, required_channels_end = FragmentHandler().get_best_12_layers(frag_id)
    required_channels = set(range(required_channels_start, required_channels_end + 1))

    # Check which slice files exist
    existing_slice_channels = set(extract_indices(frag_slice_dir, pattern=r'(\d+).tif'))
    # Check if any are missing
    missing_slice_channels = required_channels - existing_slice_channels

    if missing_slice_channels:
        errors.append(
            f"\033[91mReason:\t\tSlice channel files {format_ranges(sorted(list(missing_slice_channels)))} not found\033[0m")

    # Of those existing, get only those that we need
    valid_channels = existing_slice_channels.intersection(required_channels)

    return errors, sorted(list(valid_channels))

def find_consecutive_ch_blocks_of_size(channels, ch_block_size):
    channels = sorted(channels)
    result = []
    i = 0
    while i <= len(channels) - ch_block_size:
        if all(channels[i + j] + 1 == channels[i + j + 1] for j in range(ch_block_size - 1)):
            result.extend(channels[i:i + ch_block_size])
            i += ch_block_size  # Skip to the element after the current block
        else:
            i += 1
    return set(result)



def extract_indices(directory, pattern):
    indices = []

    for filename in os.listdir(directory):
        match = re.match(pattern, filename)
        if match:
            groups = match.groups()
            if len(groups) == 1:
                # Single number (e.g., \d+.tif)
                number = int(groups[0])
                indices.append(number)
            elif len(groups) == 2:
                # Range of numbers (e.g., inklabels_(\d+)_(\d+).png)
                start_layer, end_layer = map(int, groups)
                indices.extend(range(start_layer, end_layer + 1))

    indices = list(set(indices))  # Remove duplicates if any
    indices.sort()  # Sort the indices in ascending order
    return indices


def format_ranges(numbers, file_ending=".tif", digits=5):
    """Convert a list of numbers into a string of ranges."""
    if not numbers:
        return ""

    ranges = []
    start = end = numbers[0]

    for n in numbers[1:]:
        if n - 1 == end:  # Part of the range
            end = n
        else:  # New range
            ranges.append((start, end))
            start = end = n
    ranges.append((start, end))

    return ', '.join(
        [f"{s:0{digits}d}{file_ending} - {e:0{digits}d}{file_ending}" if s != e else f"{s:0{digits}d}.tif" for s, e in
         ranges])
