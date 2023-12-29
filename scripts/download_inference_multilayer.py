import argparse
import os
import shutil
import subprocess
import sys
from difflib import get_close_matches

from utility.fragments import get_frag_name_from_id, FragmentHandler


def validate_folder(local_path, model_run_dir):
    try:
        model_dirs = [_dir for _dir in os.listdir(local_path) if model_run_dir in _dir]
        if len(model_dirs) != 1:
            print_colored(f"Error: Expected 1 directory named {model_run_dir}, found {len(model_dirs)}", "red")
            return False

        path = os.path.join(local_path, model_dirs[0])
        files = [f for f in os.listdir(path) if f.endswith('.npy')]

        file_sizes = [os.path.getsize(os.path.join(path, f)) for f in files]
        if len(set(file_sizes)) != 1:
            print_colored("Error: Not all files in the model directory have the same size", "red")
            return False

        return True

    except Exception as e:
        print_colored(f"Error: {str(e)}", "red")
        return False


def print_colored(message, color, _print=True):
    colors = {
        "blue": '\033[94m',
        "green": '\033[92m',
        "red": '\033[91m',
        "purple": '\033[95m',
        "end": '\033[0m',
    }
    if _print:
        print(f"{colors[color]}{message}{colors['end']}")
    else:
        return f"{colors[color]}{message}{colors['end']}"


def closest_match(input_id, all_ids):
    # Finds the closest match using difflib's get_close_matches
    matches = get_close_matches(input_id, all_ids, n=1, cutoff=0.1)
    return matches[0] if matches else None


def find_directory_on_remote(server_path, fragment_id, model_run_dir):
    model_run_dir = '-'.join(model_run_dir.split('-')[:3])
    full_server_path = f"{server_path}/fragment{fragment_id}"

    return os.path.join(full_server_path, f"*{model_run_dir}*"), model_run_dir




def get_server_path(hostname, result_dir):
    server_paths = {
        "vast": f"~/kaggle1stReimp/inference/{result_dir}",
        "andro": f"~/kaggle1stReimp/inference/{result_dir}"
    }

    if hostname not in server_paths:
        print_colored("Wrong hostname", "red")
        sys.exit(1)

    return server_paths[hostname]


def get_inference_result(fragment_id, full_model_run_dir, hostname, single, force=False):
    result_dir = "single_results" if single else "results"
    server_path = get_server_path(hostname, result_dir)

    fragment_ids = FragmentHandler().get_ids()
    frag_name = get_frag_name_from_id(fragment_id).upper()

    if fragment_id not in fragment_ids:
        print_colored(f"Fragment ID not known by fragment handler: {fragment_id}", "red")

        suggestion = closest_match(fragment_id, fragment_ids)
        if suggestion:
            print_colored(f"Did you mean: {suggestion}?", "blue")
        choice = input("[y / n]")
        if choice == "y":
            fragment_id = suggestion
        else:
            exit()

    full_server_path, short_model_run_dir = find_directory_on_remote(
        fragment_id=fragment_id,
        server_path=server_path,
        model_run_dir=full_model_run_dir,
    )

    local_path = os.path.join(os.getcwd(), f"inference/{result_dir}/fragment{fragment_id}")
    os.makedirs(local_path, exist_ok=True)

    model_dirs = [_dir for _dir in os.listdir(local_path) if short_model_run_dir in _dir]

    if len(model_dirs) > 1:
        print_colored(f"ERROR:\tMore than one inference directory found: {short_model_run_dir}", "red")
        return

    if len(model_dirs) == 1:
        if validate_folder(local_path, short_model_run_dir):
            print(os.path.join(local_path, short_model_run_dir))

            if not force:
                frag_name = get_frag_name_from_id(fragment_id).upper()
                print_colored(
                    f"SKIP:\tInference download for {short_model_run_dir.upper()} and {frag_name} '{fragment_id}' already exists",
                    "blue")
                return

    if force:
        force_str = print_colored("FORCED INFERENCE DOWNLOAD:", color="purple", _print=False)
        print_colored(f"{force_str:38}{short_model_run_dir.upper()} and {frag_name} '{fragment_id}'", "green")

        # Define paths for the snapshots directory and a temporary backup location.
        snapshots_dir = os.path.join(local_path, short_model_run_dir, 'snapshots')
        temp_snapshots_dir = "/tmp/snapshots_backup"

        # Check if snapshots directory exists and move it to a temporary location.
        if os.path.isdir(snapshots_dir):
            shutil.move(snapshots_dir, temp_snapshots_dir)

        # Delete the existing local target directory.
        target_dir = os.path.join(local_path, short_model_run_dir)
        if os.path.isdir(target_dir):
            shutil.rmtree(target_dir)

        # Force download the inference using scp or similar method.
        command = f'scp -r "{hostname}:{full_server_path}" "{local_path}"'
        process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        process.wait()

        # After download, move the snapshots back if it was previously moved.
        if os.path.isdir(temp_snapshots_dir):
            shutil.move(temp_snapshots_dir, snapshots_dir)

    else:
        print_colored(f"START:\tInference download for {short_model_run_dir.upper()} "
                      f"and {frag_name} '{fragment_id}'", "green")

        # Using scp to copy the directory
        command = f'scp -r "{hostname}:{full_server_path}" "{local_path}"'
        process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        process.wait()

    # Check if scp succeeded
    if process.returncode != 0:
        print_colored("ERROR:\tError in downloading inference", "red")
        print_colored("ERROR:\tMake sure the checkpoint is correct", "red")
        print_colored("ERROR:\tMake sure that your inference is not single layer", "red")
        print_colored("ERROR:\tMake sure that your port-forwarding is activated", "red")

    print_colored(
        f"END:\tInference download for {short_model_run_dir.upper()} and {frag_name} '{fragment_id}'\n",
        "green")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download inference result from a node")
    parser.add_argument("fragment_id", type=str, help="Fragment ID")
    parser.add_argument("checkpoint_folder_name", type=str, help="Checkpoint folder name")
    parser.add_argument("hostname", type=str, help="Hostname")
    parser.add_argument("--force", action='store_true', help="Force the download and overwrite npy files")
    args = parser.parse_args()

    get_inference_result(args.fragment_id, args.checkpoint_folder_name, args.hostname, single=False, force=args.force)
