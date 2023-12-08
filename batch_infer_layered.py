import subprocess
import sys

from constants import get_ckpt_name_from_id, get_frag_name_from_id, LIVELY_MEADOW, \
    ULTRA_MAGNUS_FRAG_ID, OPTIMUS_FRAG_ID, MEGATRON_FRAG_ID, STARSCREAM_FRAG_ID, SUNSTREAKER_FRAG_ID, SOUNDWAVE_FRAG_ID, \
    IRONHIDE_FRAG_ID, RATCHET_FRAG_ID, JAZZ_FRAG_ID, DEVASTATOR_FRAG_ID, JETFIRE_FRAG_ID
from util.batch_download_frags import batch_download_frags


def print_colored(message, color):
    colors = {
        "blue": '\033[94m',
        "green": '\033[92m',
        "red": '\033[91m',
        "end": '\033[0m',
    }
    print(f"{colors[color]}{message}{colors['end']}")


FRAGMENT_IDS = [STARSCREAM_FRAG_ID, MEGATRON_FRAG_ID, IRONHIDE_FRAG_ID, RATCHET_FRAG_ID, SOUNDWAVE_FRAG_ID,
                OPTIMUS_FRAG_ID, ULTRA_MAGNUS_FRAG_ID, JAZZ_FRAG_ID, JETFIRE_FRAG_ID, SUNSTREAKER_FRAG_ID,
                DEVASTATOR_FRAG_ID]
CHECKPOINTS = [LIVELY_MEADOW]

start_idx = 0
end_idx = 60
batch_size = 32
labels = True

labels_str = ''
if labels:
    labels_str = '--labels'

    # Make sure that all fragments TIF files are existent
    batch_download_frags(FRAGMENT_IDS)

for fragment_id in FRAGMENT_IDS:
    for checkpoint in CHECKPOINTS:

        command = [
            sys.executable, 'infer_layered.py',
            str(checkpoint),
            str(fragment_id),
            '--start_idx', str(start_idx),
            '--end_idx', str(end_idx),
            '--batch_size', str(batch_size),
            str(labels_str)
        ]

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
