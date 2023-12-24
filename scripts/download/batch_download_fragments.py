import os
import re
import subprocess
import sys

from utility import AlphaBetaMeta
from utility import FragmentHandler

download_script = "scripts/download/download_slices.sh"


def determine_slice_range(fragment_id, single_layer):
    file_pattern = re.compile(r'_([0-9]+)\.png') if single_layer else re.compile(r'_([0-9]+)_([0-9]+)\.png')
    min_slice = 99999
    max_slice = 0

    label_dir = AlphaBetaMeta().get_current_binarized_label_dir(single=single_layer)

    try:
        files = os.listdir(f"{label_dir}/{fragment_id}")
        for file in files:
            match = file_pattern.search(file)
            if match:
                if single_layer:
                    slice_num = int(match.group(1))
                    min_slice = min(min_slice, slice_num)
                    max_slice = max(max_slice, slice_num)
                else:
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
            missing_slices.append(i)
        else:
            existing_slices.append(i)
            if first_slice_size is None:
                first_slice_size = os.path.getsize(slice_file)
            slice_sizes[i] = os.path.getsize(slice_file)

    for slice in existing_slices:
        if slice_sizes[slice] != first_slice_size:
            print(f"File size mismatch in slice file: {fragment_dir}/{slice:05d}.tif - {slice_sizes[slice]}",
                  file=sys.stderr)

    if not missing_slices:
        return None
    else:
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




def batch_download_frags(frag_list, consider_labels=True, single_layer=False):
    for fragment_id in frag_list:
        start_slice, end_slice = FragmentHandler().get_center_layers(frag_id=fragment_id)

        if consider_labels:
            start_slice, end_slice = determine_slice_range(fragment_id, single_layer=single_layer)

        if start_slice == 99999 or end_slice == 0:
            print(f"Fragment ID: {fragment_id}\tNo labels found")
            continue

        missing_slices = check_downloaded_slices(fragment_id, start_slice, end_slice)
        if not missing_slices:
            print(f"Fragment ID: {fragment_id}\tAll required slices found: [{start_slice}, {end_slice}]")
            continue
        else:
            ranges = get_consecutive_ranges(missing_slices)
            print(f"Fragment ID: {fragment_id}\tDownloading Slices = [{start_slice}, {end_slice}]")
            str_args = ",".join(ranges)
            command = ['bash', download_script, fragment_id, str_args]
            subprocess.run(command)
