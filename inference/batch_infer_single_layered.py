import subprocess
import sys

from scripts.batch_download_frags import batch_download_frags
from utility import AlphaBetaMeta
from utility.constants import JETFIRE_FRAG_ID, GRIMLARGE_FRAG_ID, get_ckpt_name_from_id, get_frag_name_from_id


def print_colored(message, color):
    colors = {
        "blue": '\033[94m',
        "green": '\033[92m',
        "red": '\033[91m',
        "end": '\033[0m',
    }
    print(f"{colors[color]}{message}{colors['end']}")


FRAGMENT_IDS = [JETFIRE_FRAG_ID, GRIMLARGE_FRAG_ID]
model = AlphaBetaMeta().get_previous_model()

print("Fragments for inference:\t", ", ".join(FRAGMENT_IDS))
print("Model for inference:\t", model)
CHECKPOINTS = [model]

labels = True
boost_threshold = False
verbose = True

if labels:
    # Make sure that all fragments TIF files are existent
    batch_download_frags(FRAGMENT_IDS, consider_label_files=False)

for fragment_id in FRAGMENT_IDS:
    for checkpoint in CHECKPOINTS:

        command = [
            sys.executable, 'inference/infer_single_layer.py',
            str(checkpoint),
            str(fragment_id),
        ]

        if labels:
            command.append('--labels')

        if boost_threshold:
            command.append('--boost_threshold')

        if verbose:
            command.append('--v')

        ckpt_str = get_ckpt_name_from_id(checkpoint).upper()
        frag_str = get_frag_name_from_id(fragment_id).upper()
        print_colored(f"INFERENCE:\tSTARTED:\t{ckpt_str} -> {frag_str}", "blue")

        # Execute the command
        process = subprocess.run(command, text=True)

        if process.stdout:
            print(process.stdout)

        # Check if an error occurred
        if process.returncode != 0:
            print_colored(f"INFERENCE:\tERROR:\t{ckpt_str} -> {frag_str}", "red")
            print_colored(process.stderr, "red")
        else:
            print_colored(f"INFERENCE:\tDONE:\t\t{ckpt_str} -> {frag_str}", "green")

print_colored("\nINFERENCE: COMPLETED", "green")
