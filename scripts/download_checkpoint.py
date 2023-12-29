import os
import subprocess
import argparse
import sys

GREEN = "\033[92m"
RESET = "\033[0m"


def download_checkpoint(checkpoint_name, hostname):
    checkpoint_dir = os.path.join("checkpoints", checkpoint_name)
    if os.path.isdir(checkpoint_dir):
        print(f"{GREEN}Checkpoint folder already exists. Exiting to prevent overwrite.{RESET}")
        sys.exit()

    print(f"{GREEN}Download process initiated!{RESET}")

    full_server_path = "~/kaggle1stReimp/checkpoints"
    command = f'scp -r "{hostname}:{full_server_path}/{checkpoint_name}" "checkpoints"'
    subprocess.run(command)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Download a checkpoint.")
    parser.add_argument('checkpoint_name', help="Name of the checkpoint to download")
    parser.add_argument('hostname', help="The hostname to download froam")
    args = parser.parse_args()

    download_checkpoint(args.checkpoint_name, args.hostname)
