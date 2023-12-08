import os
import re
import subprocess
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from constants import FRAGMENTS

download_script = "./util/download.sh"
label_files_dir = "./data/base_label_files/layered"


def get_fragment_ids():
    return list(FRAGMENTS.values())


def determine_slice_range(fragment_id):
    file_pattern = re.compile(r'_([0-9]+)_([0-9]+)\.png')
    min_slice = 99999
    max_slice = 0

    try:
        files = os.listdir(f"{label_files_dir}/{fragment_id}")
        for file in files:
            match = file_pattern.search(file)
            if match:
                start, end = map(int, match.groups())
                min_slice = min(min_slice, start)
                max_slice = max(max_slice, end)
    except FileNotFoundError:
        pass

    return min_slice, max_slice


def check_downloaded_slices(fragment_id, start_slice, end_slice):
    fragment_dir = f"./data/fragments/fragment{fragment_id}/slices"
    missing_slices = []
    existing_slices = []
    slice_sizes = {}
    first_slice_size = None

    if not os.path.isdir(fragment_dir):
        print(f"Slices directory does not exist for fragment ID: {fragment_id}", file=sys.stderr)
        os.makedirs(fragment_dir)

    for i in range(start_slice, end_slice + 1):
        slice_file = f"{fragment_dir}/{i:05d}.tif"
        if not os.path.isfile(slice_file):
            print(f"Missing slice file: {slice_file}", file=sys.stderr)
            missing_slices.append(i)
        else:
            existing_slices.append(i)
            if first_slice_size is None:
                first_slice_size = os.path.getsize(slice_file)
            slice_sizes[i] = os.path.getsize(slice_file)

    for slice in existing_slices:
        if slice_sizes[slice] != first_slice_size:
            print(f"File size mismatch in slice file: {fragment_dir}/{slice:05d}.tif", file=sys.stderr)

    if not missing_slices:
        return None
    else:
        print(f"Missing slices for {fragment_id}: {' '.join(map(str, missing_slices))}", file=sys.stderr)
        print(f"Existing slices for {fragment_id}: {' '.join(map(str, existing_slices))}", file=sys.stderr)
        return missing_slices


def get_consecutive_ranges(missing_slices):
    if not missing_slices:
        return []
    ranges = []
    start = end = missing_slices[0]

    for i in missing_slices[1:]:
        if i == end + 1:
            end = i
        else:
            ranges.append(f"{start:05d} {end:05d}")
            start = end = i
    ranges.append(f"{start:05d} {end:05d}")

    return ranges


def batch_download_frags(frag_list, include_labels=True):
    for fragment_id in frag_list:
        print(f"\nFragment ID: {fragment_id}")

        start_slice = 0
        end_slice = 63
        if include_labels:
            start_slice, end_slice = determine_slice_range(fragment_id)

        if start_slice == 99999 or end_slice == 0:
            print("No label files found -> Skipping download")
            continue
        else:
            print(f"Labels = [{start_slice}, {end_slice}] found")

        missing_slices = check_downloaded_slices(fragment_id, start_slice, end_slice)
        if not missing_slices:
            print("No missing slices found -> Skipping download")
            continue
        else:
            ranges = get_consecutive_ranges(missing_slices)
            print(f"Downloading missing Slices = {missing_slices}")
            str_args = ",".join(ranges)
            command = ['bash', download_script, fragment_id, str_args]
            subprocess.run(command)

    print("Batch download script execution completed.")


if __name__ == '__main__':
    fragment_list = get_fragment_ids()

    batch_download_frags(fragment_list)
