import argparse
import os
import subprocess

from utility.fragments import get_frag_name_from_id


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


def print_colored(message, color):
    colors = {
        "blue": '\033[94m',
        "green": '\033[92m',
        "red": '\033[91m',
        "end": '\033[0m',
    }
    print(f"{colors[color]}{message}{colors['end']}")


def find_directory_on_remote(server_path, fragment_id, model_run_dir):
    model_run_dir = '-'.join(model_run_dir.split('-')[:3])
    full_server_path = f"{server_path}/fragment{fragment_id}"

    return os.path.join(full_server_path, f"*{model_run_dir}*"), model_run_dir


def get_inference_result(fragment_id, full_model_run_dir, hostname, single):
    result_dir = "single_results" if single else "results"

    server_paths = {
        "vast": f"~/kaggle1stReimp/inference/{result_dir}",
        "andro": f"~/kaggle1stReimp/inference/{result_dir}"
    }

    if hostname not in server_paths:
        print_colored("Wrong hostname", "red")
        return

    server_path = server_paths[hostname]

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

            frag_name = get_frag_name_from_id(fragment_id).upper()
            print_colored(
                f"SKIP:\tInference download for {short_model_run_dir.upper()} and {frag_name} '{fragment_id}' already exists",
                "blue")
            return

    frag_name = get_frag_name_from_id(fragment_id).upper()
    print_colored(
        f"START:\tInference download for {short_model_run_dir.upper()} and {frag_name} '{fragment_id}'",
        "green")

    # Using scp to copy the directory
    command = f'scp -r "{hostname}:{full_server_path}" "{local_path}"'
    print(command)

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
    args = parser.parse_args()

    get_inference_result(args.fragment_id, args.checkpoint_folder_name, args.hostname, single=False)
