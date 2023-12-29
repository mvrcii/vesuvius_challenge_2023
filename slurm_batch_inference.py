import subprocess
import sys

from utility.fragments import *


def main():
    available_nodes = [2]
    excluded_gpus_node_one = {1, 3, 5}  # Exclude reserved-164-01 gpus here
    excluded_gpus_node_two = {0, 1, 2, 3, 4, 7}         # Exclude reserved-237-02 gpus here

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
        SUNSTREAKER_FRAG_ID, BLASTER_FRAG_ID
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

        try:
            subprocess.run(command)
            print(f"Job for {get_frag_name_from_id(frag_id):17} {frag_id:17} queued on Node={node_id} and GPU={gpu_id}")
        except Exception as e:
            print(f"Exception occurred while queuing {frag_id}: {e}")

    print("Batch job submission completed.")


if __name__ == "__main__":
    main()
