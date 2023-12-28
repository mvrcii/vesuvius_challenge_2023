import argparse
import re
import subprocess
import time


def main():
    parser = argparse.ArgumentParser(description="Submit a create dataset job to Slurm.")
    parser.add_argument("config_path", type=str, help="Path to the config file.")
    parser.add_argument('--node2', action='store_true', help='Perform advanced TTA')
    args = parser.parse_args()

    node_name = "tenant-ac-nowak-h100-reserved-237-02" if args.node2 else "tenant-ac-nowak-h100-reserved-164-01"

    cmd_str = (f"python3 "
               f"multilayer_approach/create_dataset_segmentation.py {args.config_path}")

    slurm_cmd = f'sbatch --nodelist={node_name} --wrap="{cmd_str}" -o "logs/slurm-%j.out"'

    # Run the sbatch command and capture its output
    result = subprocess.run(slurm_cmd, shell=True, capture_output=True, text=True)

    # Extract job ID from the output
    match = re.search(r"Submitted batch job (\d+)", result.stdout)
    if match:
        job_id = match.group(1)
        print(f"Slurm job ID: {job_id}")

        delay_seconds = 2  # Adjust this value as needed
        print(f"Waiting for {delay_seconds} seconds before tailing the log file...")
        time.sleep(delay_seconds)

        tail_cmd = f"tail -f logs/slurm-{job_id}.out"
        subprocess.run(tail_cmd, shell=True)
    else:
        print("Failed to submit job to Slurm or parse job ID.")

if __name__ == "__main__":
    main()
