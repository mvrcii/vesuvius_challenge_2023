import argparse
import subprocess


def main():
    parser = argparse.ArgumentParser(description="Submit a training job to Slurm.")
    parser.add_argument("config_path", type=str, help="Path to the config file.")
    parser.add_argument('--seed', type=int, default=None, help='Optional seed for the script')

    args = parser.parse_args()

    seed_str = f"--seed {args.seed}" if args.seed else ""
    cmd_str = f"python3 train.py {args.config_path} {seed_str}"

    slurm_cmd = f'srun \
    --wrap="{cmd_str}" \
    -o "logs/slurm-%j.out"'

    subprocess.run(slurm_cmd, shell=True)


if __name__ == "__main__":
    main()
