import argparse
import re
import subprocess
import time


def main():
    parser = argparse.ArgumentParser(description="Submit a training job to Slurm.")
    parser.add_argument("config_path", type=str, help="Path to the config file.")
    parser.add_argument('fragment_id', type=str, help='The fragment to infer.')
    parser.add_argument('--stride', type=int, default=2, help='Stride (default: 2)')
    parser.add_argument('--gpu', type=int, default=0, help='GPU (default: 0)')
    parser.add_argument('--tta', action='store_true', help='Perform advanced TTA')
    args = parser.parse_args()

    tta_str = "_tta" if args.tta else ""

    cmd_str = (f"python3 "
               f"multilayer_approach/infer_layered_segmentation_padto16{tta_str}.py "
               f"{args.config_path} {args.fragment_id} --stride {args.stride} --gpu {args.gpu}")

    slurm_cmd = f'sbatch --wrap="{cmd_str}" -o "logs/slurm-%j.out"'

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
