import subprocess
import sys

from constants import BLASTER_FRAG_ID
from constants import STELLAR_VIOLET, REVIVED_BEE, AMBER_PLANT, CHOCOLATE_FOG


FRAGMENT_IDS = [BLASTER_FRAG_ID]
CHECKPOINTS = [AMBER_PLANT, CHOCOLATE_FOG, REVIVED_BEE, STELLAR_VIOLET]

start_idx = 0
end_idx = 62
batch_size = 16

for fragment_id in FRAGMENT_IDS:
    for checkpoint in CHECKPOINTS:

        command = [
            sys.executable, 'infer_layered.py',
            '--fragment_id', str(fragment_id),
            '--checkpoint_folder_name', str(checkpoint),
            '--start_idx', str(start_idx),
            '--end_idx', str(end_idx),
            '--batch_size', str(batch_size)
        ]

        # Execute the command
        process = subprocess.run(command, capture_output=True, text=True)

        # Check if an error occurred
        if process.returncode != 0:
            print(f"Error occurred while processing fragment {fragment_id} with checkpoint {checkpoint}:")
            print(process.stderr)

print("Batch inference completed.")
