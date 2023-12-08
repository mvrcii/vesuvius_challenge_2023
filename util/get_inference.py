import os
import subprocess
import argparse
from tqdm import tqdm


def print_colored(message, color):
    colors = {
        "blue": '\033[94m',
        "green": '\033[92m',
        "red": '\033[91m',
        "end": '\033[0m',
    }
    print(f"{colors[color]}{message}{colors['end']}")


def main(fragment_id, checkpoint_folder_name, hostname):
    server_paths = {
        "vast": "~/kaggle1stReimp/inference/results",
        "slurm": "/scratch/medfm/vesuv/kaggle1stReimp/inference/results",
    }

    if hostname not in server_paths:
        print_colored("Wrong hostname", "red")
        return

    server_path = server_paths[hostname]
    full_server_path = f"{server_path}/fragment{fragment_id}/{checkpoint_folder_name}"
    local_path = os.path.join(os.getcwd(), f"inference/results/fragment{fragment_id}")

    # Check if the directory exists, create it if not
    if not os.path.isdir(local_path):
        os.makedirs(local_path)

    # Using scp to copy the directory
    command = f"scp -r {hostname}:{full_server_path} {local_path}"

    print_colored("Starting folder copy...", "blue")

    # Execute scp command
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    for _ in tqdm(process.stdout):
        pass  # This will show the progress of the copy

    # Wait for the process to complete
    process.wait()

    # Check if scp succeeded
    if process.returncode == 0:
        print_colored("Folder copied successfully.", "green")
    else:
        print_colored("Error in copying folder.", "red")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Copy folder from server script")
    parser.add_argument("fragment_id", type=str, help="Fragment ID")
    parser.add_argument("checkpoint_folder_name", type=str, help="Checkpoint folder name")
    parser.add_argument("hostname", type=str, help="Hostname")
    args = parser.parse_args()

    main(args.fragment_id, args.checkpoint_folder_name, args.hostname)