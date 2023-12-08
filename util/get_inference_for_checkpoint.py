import subprocess

from constants import *


def get_inference(frag_ids, checkpoint_path, host):
    for frag_id in frag_ids:
        command = [
            'bash',
            "./util/get_inference.sh",
            frag_id,
            checkpoint_path,
            host
        ]

        # Execute the command
        process = subprocess.run(command, text=True)

        if process.stdout:
            print(process.stdout)


if __name__ == '__main__':
    fragments = FRAGMENTS_ALPHA
    checkpoint = LIVELY_MEADOW

    get_inference(frag_ids=fragments, checkpoint_path=checkpoint, host="vast")
