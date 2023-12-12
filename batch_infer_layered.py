import subprocess
import sys

from constants import *
from meta import AlphaBetaMeta
from util.batch_download_frags import batch_download_frags


def print_colored(message, color):
    colors = {
        "blue": '\033[94m',
        "green": '\033[92m',
        "red": '\033[91m',
        "end": '\033[0m',
    }
    print(f"{colors[color]}{message}{colors['end']}")

BOOST_EXP_FRAG_IDS = [HOT_ROD_FRAG_ID, THUNDERCRACKER_FRAG_ID, SUNSTREAKER_FRAG_ID, BLASTER_FRAG_ID, JAZZILLA_FRAG_ID]
model = DEFT_YOGURT

# BOOST_EXP_FRAG_IDS = [DEVASTATOR_FRAG_ID, ULTRA_MAGNUS_FRAG_ID, IRONHIDE_FRAG_ID]
# model = UPBEAT_TREE

FRAGMENT_IDS = BOOST_EXP_FRAG_IDS
# FRAGMENT_IDS = AlphaBetaMeta().get_current_inference_fragments()
# model = AlphaBetaMeta().get_current_model()


print("Fragments for inference:\t", ", ".join(FRAGMENT_IDS))
print("Model for inference:\t", model)
CHECKPOINTS = [model]

start_idx = 0
end_idx = 60
batch_size = 32
labels = True
boosted_threshold = True
verbose = False

if labels:
    # Make sure that all fragments TIF files are existent
    batch_download_frags(FRAGMENT_IDS, consider_label_files=False)

for fragment_id in FRAGMENT_IDS:
    for checkpoint in CHECKPOINTS:

        command = [
            sys.executable, 'infer_layered.py',
            str(checkpoint),
            str(fragment_id),
            '--start_idx', str(start_idx),
            '--end_idx', str(end_idx),
            '--batch_size', str(batch_size),
        ]

        if labels:
            command.append('--labels')

        if boosted_threshold:
            command.append('--boosted_threshold')

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
