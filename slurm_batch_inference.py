import re
import subprocess
import sys

from utility.fragments import *


def process_output(frag_id, command, tta, stride, checkpoint):
    try:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        stdout, _ = process.communicate()

        match = re.search(r"Slurm job ID: (\d+)", stdout)
        if match:
            job_id = match.group(1)

            # Build the string based on TTA, checkpoint, and stride
            tta_str = " + TTA" if tta else ""
            checkpoint_name = checkpoint.split('-')[0]  # Assuming checkpoint format includes name
            stride_str = f"S{stride}"
            print(f"{get_frag_name_from_id(frag_id)} {stride_str}{tta_str} ({checkpoint_name}) {job_id}")
        else:
            print(f"Failed to get job ID for {frag_id}")

    except Exception as e:
        print(f"Exception occurred while starting {frag_id}: {e}")



def main():
    available_nodes = [2]
    excluded_gpus_node_one = {1, 3, 5}  # Exclude reserved-164-01 gpus here
    excluded_gpus_node_two = {0, 1, 3, 4, 5, 6, 7}  # Exclude reserved-237-02 gpus here

    available_gpus = [gpu_id for gpu_id in range(0, 8)]
    available_gpu_combinations = [(node_id, gpu_id) for node_id in available_nodes for gpu_id in available_gpus
                                  if gpu_id not in excluded_gpus_node_one and node_id == 1
                                  or gpu_id not in excluded_gpus_node_two and node_id == 2]

    tta = True
    stride = 2
    checkpoint = "efficient-aardvark-1173-unetr-sf-b5-231229-082126"

    # frags_2_infer = [
    #     BLASTER_FRAG_ID, HOT_ROD_FRAG_ID, ULTRA_MAGNUS_FRAG_ID, DEVASBIGGER_FRAG_ID, SKYHUGE_FRAG_ID,
    #     IRONHIDE_FRAG_ID, GRIMHUGE_FRAG_ID, JAZZBIGGER_FRAG_ID, SUNSTREAKER_FRAG_ID, THUNDERCRACKER_FRAG_ID,
    #     BLUEBIGGER_FRAG_ID, TRAILBREAKER_FRAG_ID,
    # ]

    frags_2_infer = [
        IRONHIDE_FRAG_ID, ULTRA_MAGNUS_FRAG_ID
    ]

    for frag_id, (node_id, gpu_id) in zip(frags_2_infer, available_gpu_combinations):
        command = [sys.executable, "slurm_inference.py",
                   str(checkpoint),
                   str(frag_id),
                   '--gpu', str(gpu_id),
                   '--stride', str(stride)]

        if node_id == 2:
            command.append('--node2')

        if tta:
            command.append('--tta')

        if tta:
            command.append('--no_tail')

        try:
            result = subprocess.run(command, capture_output=True, text=True)
            print(result)
            match = re.search(r"Slurm job ID: (\d+)", result.stdout)
            if match:
                job_id = match.group(1)

                # Build the string based on TTA, checkpoint, and stride
                tta_str = " + TTA" if tta else ""
                checkpoint_name = checkpoint.split('-')[0]  # Assuming checkpoint format includes name
                stride_str = f"S{stride}"
                print(f"{get_frag_name_from_id(frag_id)} {stride_str}{tta_str} ({checkpoint_name}) {job_id}")
            else:
                print(f"Failed to get job ID for {frag_id}")

        except Exception as e:
            print(f"Exception occurred while starting {frag_id}: {e}")

print("Batch job submission completed.")

if __name__ == "__main__":
    main()
