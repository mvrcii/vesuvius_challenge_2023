import subprocess
import sys

from constants import BLASTER_FRAG_ID, get_ckpt_name_from_id, get_frag_name_from_id
from constants import STELLAR_VIOLET, REVIVED_BEE, AMBER_PLANT, CHOCOLATE_FOG


def print_colored(message, color):
    colors = {
        "blue": '\033[94m',
        "green": '\033[92m',
        "red": '\033[91m',
        "end": '\033[0m',
    }
    print(f"{colors[color]}{message}{colors['end']}")


FRAGMENT_IDS = [BLASTER_FRAG_ID]
CHECKPOINTS = [AMBER_PLANT, CHOCOLATE_FOG, REVIVED_BEE, STELLAR_VIOLET]

start_idx = 0
end_idx = 60
batch_size = 16

for fragment_id in FRAGMENT_IDS:
    for checkpoint in CHECKPOINTS:

        command = [
            sys.executable, 'infer_layered.py',
            str(checkpoint),
            str(fragment_id),
            '--start_idx', str(start_idx),
            '--end_idx', str(end_idx),
            '--batch_size', str(batch_size)
        ]

        ckpt_str = get_ckpt_name_from_id(checkpoint).upper()
        frag_str = get_frag_name_from_id(fragment_id).upper()
        print_colored(f"INFERENCE:\t\tSTARTED:\t\t{ckpt_str} -> {frag_str}", "blue")

        # Execute the command
        process = subprocess.run(command, text=True)

        print(process.stdout)

        # Check if an error occurred
        if process.returncode != 0:
            print_colored(f"INFERENCE:\t\tERROR:\t\t{ckpt_str} -> {frag_str}", "red")
            print_colored(process.stderr, "red")
        else:
            print_colored(f"INFERENCE:\t\tDONE:\t\t{ckpt_str} -> {frag_str}", "green")

print_colored("\nINFERENCE: COMPLETED", "green")
