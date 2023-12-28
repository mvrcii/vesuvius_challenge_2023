import argparse
import re
import subprocess
import time


def main():
    parser = argparse.ArgumentParser(description="Submit a training job to Slurm.")
    parser.add_argument("config_path", type=str, help="Path to the config file.")
    parser.add_argument('--seed', type=int, default=None, help='Optional seed for the script')

    args = parser.parse_args()

    seed_str = f"--seed {args.seed}" if args.seed else ""
    cmd_str = f"python3 train.py {args.config_path} {seed_str}"

    slurm_cmd = f'sbatch --wrap="{cmd_str}" -o "logs/slurm-%j.out"'

    result = subprocess.run(slurm_cmd, shell=True, capture_output=True, text=True)

    # Extract job ID from the output
    match = re.search(r"Submitted batch job (\d+)", result.stdout)
    if match:
        job_id = match.group(1)
        print(f"Slurm job ID: {job_id}")

        time.sleep(2)

        tail_cmd = f"tail -f logs/slurm-{job_id}.out"
        subprocess.run(tail_cmd, shell=True)
    else:
        print("Failed to submit job to Slurm or parse job ID.")


if __name__ == "__main__":
    main()
