import subprocess
import sys

from constants import BLASTER_FRAG_ID, get_ckpt_name_from_id, get_frag_name_from_id
from constants import STELLAR_VIOLET, REVIVED_BEE, AMBER_PLANT, CHOCOLATE_FOG


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

        # Execute the command
        print(f"Started inference with checkpoint {get_ckpt_name_from_id(checkpoint).upper()} "
              f"on fragment {get_frag_name_from_id(fragment_id).upper()}")

        process = subprocess.run(command, text=True)

        # Print stdout and stderr
        print("Output:")
        print(process.stdout)
        print("Error (if any):")
        print(process.stderr)

        # Check if an error occurred
        if process.returncode != 0:
            print(f"Error occurred while processing fragment {fragment_id} with checkpoint {checkpoint}:")
            print(process.stderr)
        else:
            print(f"Finished inference with checkpoint {get_ckpt_name_from_id(checkpoint).upper()} "
                  f"on fragment {get_frag_name_from_id(fragment_id).upper()}")

print("Batch inference completed.")
