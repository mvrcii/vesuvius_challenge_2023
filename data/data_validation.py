import os
import re

from constants import get_frag_name_from_id, FRAGMENTS, SIDESWIPE_FRAG_ID, JETFIRE_FRAG_ID


def validate_fragments(config, label_dir):
    all_errors = []
    frag_id_2_channels = {}
    frag_id_2_existing_channels = {}

    valid_fragments = {}
    excluded_fragments = []

    for frag_id in [JETFIRE_FRAG_ID]:
        val_errors, frag_channels, existing_channels = validate_fragment_files(frag_id=frag_id, cfg=config,
                                                                               label_dir=label_dir)

        if len(frag_channels) > 0:
            frag_id_2_channels[frag_id] = frag_channels

        if len(existing_channels) > 0:
            frag_id_2_existing_channels[frag_id] = existing_channels

        frag_str = f"Fragment:\t{get_frag_name_from_id(frag_id)} ({frag_id})"
        if val_errors:
            all_errors.extend(val_errors)
            excluded_fragments.append([frag_str] + val_errors)
        else:
            valid_fragments[frag_id] = [frag_str]

    for frag_id, valid in valid_fragments.items():
        type_str = 'Images & Labels'

        new_channels = frag_id_2_channels.get(frag_id, False)
        existing_channels = frag_id_2_existing_channels.get(frag_id, False)

        if existing_channels and not new_channels:
            valid.append(
                f"Reason:\t\t{type_str} for channels {format_ranges(sorted(list(existing_channels)), '')} already exist")
            excluded_fragments.append(valid)
        elif not existing_channels and new_channels:
            valid.append(
                f"Reason:\t\t{type_str} for channels {format_ranges(sorted(list(new_channels)), '')} will be created")
            print_checks([], valid)
        elif existing_channels and new_channels:
            valid.append(
                f"Reason:\t\t{type_str} for channels {format_ranges(sorted(list(existing_channels)), '')} already exist")
            valid.append(
                f"Reason:\t\t{type_str} for channels {format_ranges(sorted(list(new_channels)), '')} will be created")
            print_checks([], valid)
        else:
            pass

    for excluded in excluded_fragments:
        print_checks(excluded, [])

    print("\n")

    return frag_id_2_channels


def print_checks(errors, valids):
    for valid in valids:
        print(f"\033[92mRequired:\t\t{valid}\033[0m")
    for error in errors:
        if "Fragment" in error:
            print(f"\033[94mExcluded:\t\t{error}\033[0m")
        else:
            print(f"\033[94mExcluded:\033[0m\t\t{error}")


def validate_fragment_files(frag_id, cfg, label_dir):
    errors = []
    frag_dir = os.path.join(cfg.work_dir, "data", "fragments", f"fragment{frag_id}")
    frag_label_dir = os.path.join(label_dir, f"{frag_id}")

    errors.extend(validate_fragment_dir(frag_dir))

    valid_errors, valid_channels, existing_channels = validate_labels(cfg, frag_id, frag_dir, frag_label_dir)
    errors.extend(valid_errors)

    errors.extend(validate_masks(frag_dir))

    return errors, valid_channels, existing_channels


def validate_fragment_dir(frag_dir):
    if not os.path.isdir(frag_dir):
        return [f"\033[91mReason:\t\tFragment directory '{frag_dir}' does not exist\033[0m"]

    return []


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


def validate_labels(cfg, frag_id, frag_dir, label_dir):
    errors = []
    slice_dir = os.path.join(frag_dir, 'slices')

    if not os.path.isdir(label_dir):
        errors.append(f"\033[91mReason:\t\tLabel directory not found\033[0m")

    if not os.path.isdir(slice_dir):
        errors.append(f"\033[91mReason:\t\tSlice directory not found\033[0m")

    valid_channels = []
    existing_channels = []

    # If no errors yet, continue
    if len(errors) == 0:
        existing_negative_channels = set(extract_indices(label_dir, pattern=r'negatives_(\d+)_(\d+).png'))
        existing_label_channels = set(extract_indices(label_dir, pattern=r'inklabels_(\d+)_(\d+).png'))

        existing_negative_channels = find_consecutive_ch_blocks_of_size(list(existing_negative_channels), cfg.in_chans)
        existing_label_channels = find_consecutive_ch_blocks_of_size(list(existing_label_channels), cfg.in_chans)

        required_channels = existing_label_channels.union(existing_negative_channels)

        existing_slice_channels = set(extract_indices(slice_dir, pattern=r'(\d+).tif'))

        # Check for already existing images / labels patches
        # start_indices = extract_indices(existing_image_dir, pattern=r".*ch(\d+)_.*\.npy$")
        # existing_channels = set([index + i for i in range(cfg.in_chans) for index in start_indices])
        # required_channels -= existing_channels

        valid_channels = existing_slice_channels.intersection(required_channels)

        missing_slice_channels = required_channels - existing_slice_channels
        if missing_slice_channels:
            errors.append(
                f"\033[91mReason:\t\tSlice channel files {format_ranges(sorted(list(missing_slice_channels)))} not found\033[0m")

    return errors, sorted(list(valid_channels)), sorted(list(existing_channels))


def validate_masks(frag_dir):
    file = 'mask.png'
    if not os.path.isfile(os.path.join(frag_dir, file)):
        return [f"\033[91mReason:\t\tMask file not found\033[0m"]

    return []


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


def format_ranges(numbers, file_ending=".tif"):
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
        [f"{s:02d}{file_ending}-{e:02d}{file_ending}" if s != e else f"{s:02d}.tif" for s, e in ranges])
