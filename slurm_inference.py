import argparse
import subprocess


def main():
    parser = argparse.ArgumentParser(description="Submit a training job to Slurm.")
    parser.add_argument("config_path", type=str, help="Path to the config file.")
    parser.add_argument('fragment_id', type=str, help='The fragment to infer.')

    args = parser.parse_args()

    cmd_str = f"python3 multilayer_approach/infer_layered_segmentation_padto16.py {args.config_path} {args.fragment_id}"

    slurm_cmd = f'sbatch --wrap="{cmd_str}" -o "logs/slurm-%j.out"'

    subprocess.run(slurm_cmd, shell=True)


if __name__ == "__main__":
    main()
