import os
import subprocess
import argparse
import sys

GREEN = "\033[92m"
RESET = "\033[0m"


def download_checkpoint(checkpoint_name):
    # Assuming there's only one hostname since the original script had a single option
    hostname = "vast"

    # Check if checkpoint already exists
    checkpoint_dir = os.path.join("checkpoints", checkpoint_name)
    if os.path.isdir(checkpoint_dir):
        print(f"{GREEN}Checkpoint folder already exists. Exiting to prevent overwrite.{RESET}")
        sys.exit()

    print(f"{GREEN}Download process initiated!{RESET}")

    full_server_path = "~/kaggle1stReimp/checkpoints"
    command = f'scp -r "{hostname}:{full_server_path}/{checkpoint_name}" "checkpoints/{checkpoint_name}"'
    subprocess.run(command)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Download a checkpoint.")
    parser.add_argument('checkpoint_name', help="Name of the checkpoint to download")
    args = parser.parse_args()

    download_checkpoint(args.checkpoint_name)
