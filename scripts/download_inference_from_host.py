import argparse
import os
import shutil
import subprocess
import sys
from difflib import SequenceMatcher

from utility.checkpoints import CHECKPOINTS, get_checkpoint_name
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


def find_directory_on_remote(server_path, fragment_id, model_run_dir):
    model_run_dir = '-'.join(model_run_dir.split('-')[:3])
    full_server_path = f"{server_path}/fragment{fragment_id}"

    return os.path.join(full_server_path, f"*{model_run_dir}*"), model_run_dir


def get_server_path(hostname, result_dir):
    server_paths = {
        "vast": f"~/MT3/inference/{result_dir}",
        "andro": f"~/MT3/inference/{result_dir}"
    }

    if hostname not in server_paths:
        print_colored("Wrong hostname", "red")
        sys.exit(1)

    return server_paths[hostname]


def similarity_score(a, b):
    return SequenceMatcher(None, a, b).ratio()


def dynamic_closest_matches(input_str, options, threshold=0.6):
    input_str = input_str.upper()
    scored_options = [(option, similarity_score(input_str, option)) for option in options]
    close_matches = [option for option, score in scored_options if score >= threshold]
    return sorted(close_matches, key=lambda x: -similarity_score(input_str, x))


def get_inference_result(fragment_id_or_name, checkpoint_keyword, hostname, force=False):
    result_dir = "results"
    server_path = get_server_path(hostname, result_dir)
    fragment_ids = FragmentHandler().get_ids()
    fragment_names = FragmentHandler().get_names()
    name_to_id = FragmentHandler().FRAGMENTS
    checkpoint_dict = CHECKPOINTS

    # Check if input is an ID or a Name and convert to ID if necessary
    if fragment_id_or_name in fragment_ids or fragment_id_or_name in name_to_id:
        fragment_id = name_to_id.get(fragment_id_or_name, fragment_id_or_name)
    else:
        id_suggestions = dynamic_closest_matches(fragment_id_or_name, fragment_ids)
        name_suggestions = dynamic_closest_matches(fragment_id_or_name, fragment_names)
        suggestions = list(set(id_suggestions + name_suggestions))

        if suggestions:
            if len(suggestions) > 1:
                print("Did you mean one of the following?")
                for idx, suggestion in enumerate(suggestions, 1):
                    print(f"{idx}. {suggestion}")

                choice = input("Enter the number of the correct option, or 'n' to cancel: ")
                if choice.isdigit() and 1 <= int(choice) <= len(suggestions):
                    selected_suggestion = suggestions[int(choice) - 1]
                    print(selected_suggestion)
                    fragment_id = name_to_id.get(selected_suggestion, selected_suggestion)
                else:
                    print("Invalid selection.")
                    exit()
            elif len(suggestions) == 1:
                fragment_id = name_to_id.get(suggestions[0])
            else:
                print("No valid suggestion found.")
                exit()
        else:
            print_colored(f"No close match found for: {fragment_id_or_name}", "red")
            exit()

    # Handle checkpoint
    full_checkpoint_name = get_checkpoint_name(checkpoint_keyword, checkpoint_dict)

    frag_name = get_frag_name_from_id(fragment_id)
    full_server_path, short_model_run_dir = find_directory_on_remote(
        fragment_id=fragment_id,
        server_path=server_path,
        model_run_dir=full_checkpoint_name,  # Use the full checkpoint name here
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
    parser.add_argument("checkpoint_folder_name", type=str, help="Checkpoint folder name")
    parser.add_argument("fragment_id", type=str, help="Fragment ID")
    parser.add_argument("--hostname", type=str, default='andro', help="Hostname")
    parser.add_argument("--force", action='store_false', help="Force the download and overwrite npy files")
    args = parser.parse_args()

    get_inference_result(fragment_id_or_name=args.fragment_id,
                         checkpoint_keyword=args.checkpoint_folder_name,
                         hostname=args.hostname, force=args.force)
