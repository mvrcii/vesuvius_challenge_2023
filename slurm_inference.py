import argparse
import re
import subprocess
import time


def main():
    parser = argparse.ArgumentParser(description="Submit a inference job to Slurm.")
    parser.add_argument("checkpoint_path", type=str, help="Path to the checkpoint file.")
    parser.add_argument('fragment_id', type=str, help='The fragment to infer.')
    parser.add_argument('--stride', type=int, default=2, help='Stride (default: 2)')
    parser.add_argument('--gpu', type=int, default=0, help='GPU (default: 0)')
    parser.add_argument('--tta', action='store_true', help='Perform advanced TTA')
    parser.add_argument('--node2', action='store_true', help='Use Node 2')
    parser.add_argument('--no_tail', action='store_true', help='Tail into the inference')
    parser.add_argument('--full_sweep', action='store_true', help='Do a full layer inference sweep (0-63)')
    args = parser.parse_args()

    tta_str = "_tta" if args.tta else ""

    node_name = "tenant-ac-nowak-h100-reserved-237-02" if args.node2 else "tenant-ac-nowak-h100-reserved-164-01"
    script_name = f"multilayer_approach/infer_layered_segmentation_padto16{tta_str}.py"
    print("Using", script_name)

    command = ["python3", str(script_name), str(args.checkpoint_path), str(args.fragment_id),
               '--stride', str(args.stride),
               '--gpu', str(args.gpu)]

    if not args.tta and args.full_sweep:
        command.append('--full_sweep')

    slurm_cmd = f'sbatch --nodelist={node_name} --wrap="{" ".join(command)}" -o "logs/slurm-%j.out"'

    # Run the sbatch command and capture its output
    result = subprocess.run(slurm_cmd, shell=True, capture_output=True, text=True)

    # Extract job ID from the output
    match = re.search(r"Submitted batch job (\d+)", result.stdout)
    if match:
        job_id = match.group(1)
        print(f"Slurm job ID: {job_id}")

        if not args.no_tail:
            delay_seconds = 2  # Adjust this value as needed
            print(f"Waiting for {delay_seconds} seconds before tailing the log file...")
            time.sleep(delay_seconds)

            tail_cmd = f"tail -f logs/slurm-{job_id}.out"
            subprocess.run(tail_cmd, shell=True)
    else:
        print("Failed to submit job to Slurm or parse job ID.")

if __name__ == "__main__":
    main()
