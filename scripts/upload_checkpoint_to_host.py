import os
import subprocess
import argparse
import sys

GREEN = "\033[92m"
RESET = "\033[0m"


def download_checkpoint(checkpoint_name, hostname):
    # Check if checkpoint already exists
    checkpoint_dir = os.path.join("checkpoints", checkpoint_name)
    if os.path.isdir(checkpoint_dir):
        print(f"{GREEN}Checkpoint folder already exists. Exiting to prevent overwrite.{RESET}")
        sys.exit()

    print(f"{GREEN}Download process initiated!{RESET}")

    full_server_path = "~/MT3/checkpoints"
    command = f'scp -r "checkpoints/{checkpoint_name}" "{hostname}:{full_server_path}"'
    subprocess.run(command)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Upload a checkpoint.")
    parser.add_argument('checkpoint_name', help="Name of the checkpoint to download")
    parser.add_argument('hostname', help="Name of the checkpoint to download")
    args = parser.parse_args()

    download_checkpoint(args.checkpoint_name, args.hostname)
