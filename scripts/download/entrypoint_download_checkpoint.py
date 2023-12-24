import os
import subprocess
import sys

from scripts.download.entrypoint_download_fragments import get_user_input, get_user_string_input
from utility.checkpoints import CHECKPOINTS

GREEN = "\033[92m"
RED = "\033[91m"
BLUE = "\033[94m"
RESET = "\033[0m"


def download_checkpoint():
    download_script = "scripts/download/download_checkpoint.sh"
    hostnames = ["vast"]
    options = ["Yes", "No"]

    checkpoint_from_constants = get_user_input("Is the checkpoint contained in constants?", options)
    from_constants = (checkpoint_from_constants == 1)

    if from_constants:
        choices = [_ for _ in CHECKPOINTS.keys()]
        selected_ckpt_idx = get_user_input("Select an existing checkpoint:", choices)
        checkpoint_key = list(CHECKPOINTS.keys())[selected_ckpt_idx - 1]
        checkpoint_name = CHECKPOINTS[checkpoint_key]
    else:
        checkpoint_name = get_user_string_input("Enter a full checkpoint name:")

    if len(hostnames) == 1:
        hostname = hostnames[0]
    else:
        selected_host_idx = get_user_input("Select a host:", hostnames)
        hostname = hostnames[selected_host_idx - 1]

    # Check if checkpoint already exists
    checkpoint_dir = os.path.join("checkpoints", checkpoint_name)
    if os.path.isdir(checkpoint_dir):
        override = get_user_input("Checkpoint folder exists, do you want to override it?", options)
        if override == 2:
            sys.exit()
        else:

            print(f"\n{BLUE}Overriding Checkpoint: {RESET}{GREEN} Download process initiated!{RESET}\n")
    else:
        print("\n" + GREEN + "Download process initiated!" + RESET + "\n")

    full_server_path = "~/kaggle1stReimp/checkpoints"

    command = f'scp -r "{hostname}:{full_server_path}/{checkpoint_name}" "checkpoints/{checkpoint_name}"'
    subprocess.run(command)


if __name__ == '__main__':
    download_checkpoint()
