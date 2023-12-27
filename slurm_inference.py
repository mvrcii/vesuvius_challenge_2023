import argparse
import re
import subprocess
import time


def main():
    parser = argparse.ArgumentParser(description="Submit a training job to Slurm.")
    parser.add_argument("config_path", type=str, help="Path to the config file.")
    parser.add_argument('fragment_id', type=str, help='The fragment to infer.')
    parser.add_argument('--stride', type=int, default=2, help='Stride (default: 2)')

    args = parser.parse_args()

    cmd_str = (f"python3 "
               f"multilayer_approach/infer_layered_segmentation_padto16.py "
               f"{args.config_path} {args.fragment_id} --stride {args.stride}")

    slurm_cmd = f'sbatch --wrap="{cmd_str}" -o "logs/slurm-%j.out"'

    # Run the sbatch command and capture its output
    result = subprocess.run(slurm_cmd, shell=True, capture_output=True, text=True)

    # Extract job ID from the output
    match = re.search(r"Submitted batch job (\d+)", result.stdout)
    if match:
        job_id = match.group(1)
        print(f"Slurm job ID: {job_id}")

        delay_seconds = 5  # Adjust this value as needed
        print(f"Waiting for {delay_seconds} seconds before tailing the log file...")
        time.sleep(delay_seconds)

        # Ask user if they want to tail the log file
        user_response = input("Do you want to tail the log file? (y/n): ").strip().lower()
        if user_response == 'y':
            # Tail the Slurm job's log file
            tail_cmd = f"tail -f logs/slurm-{job_id}.out"
            print(f"Executing: {tail_cmd}")
            subprocess.run(tail_cmd, shell=True)
        else:
            print("Skipping tailing the log file.")
    else:
        print("Failed to submit job to Slurm or parse job ID.")

if __name__ == "__main__":
    main()
