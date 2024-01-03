import argparse
import os
import re
import subprocess
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utility.configs import Config
from utility.fragments import FragmentHandler, SUPERSEDED_FRAGMENTS
from utility.meta_data import AlphaBetaMeta

download_script = "./scripts/utils/download_fragment.sh"


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
    fragment_dir = f"fragments/fragment{fragment_id}/slices"
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


def download_frags(frag_list):
    for fragment_id in frag_list:
        start_slice, end_slice = FragmentHandler().get_best_12_layers(frag_id=fragment_id)

        missing_slices = check_downloaded_slices(fragment_id, start_slice, end_slice)

        if not missing_slices:
            print(f"Fragment ID: {fragment_id}\tAll required slices found: [{start_slice}, {end_slice}]")
            continue
        else:
            ranges = get_consecutive_ranges(missing_slices)
            print(f"Fragment ID: {fragment_id}\tDownloading Slices = [{start_slice}, {end_slice}]")
            str_args = ",".join(ranges)

            if fragment_id in SUPERSEDED_FRAGMENTS:
                print("Warning: Fragment superseded, added suffix for download!")
                fragment_id += "_superseded"

            command = ['bash', download_script, fragment_id, str_args]
            subprocess.run(command)


def get_sys_args():
    parser = argparse.ArgumentParser(description='Download Fragments')
    parser.add_argument('config_path', type=str, help='Path to the configuration file')
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = get_sys_args()

    config = Config.load_from_file(args.config_path)

    fragments_2_download = config.fragment_ids + config.validation_fragments
    download_frags(fragments_2_download)
