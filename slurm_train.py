import argparse
import subprocess


def main():
    parser = argparse.ArgumentParser(description="Submit a training job to Slurm.")
    parser.add_argument("--gpu_type", type=str, default="c", choices=['a', 'b', 'c', '8a'],
                        help="Type of GPU to use (a, b, c, 8a). Default is 'c'.")
    parser.add_argument("--config_path", type=str, required=True,
                        help="Path to the configuration file.")
    parser.add_argument('--seed', type=int, default=None, help='Optional seed for the script')

    args = parser.parse_args()

    # Map the GPU type to the corresponding Slurm GPU resource
    gpu_mapping = {
        "a": "rtx3090:1",
        "b": "rtx3090:1",
        "c": "rtx4090:1",
        "8a": "rtx2080ti:7"
    }

    gpu_resource = gpu_mapping[args.gpu_type]

    cmd_str = f"python train.py {args.config_path} --seed {args.seed}"

    slurm_cmd = f'sbatch -p ls6 \
    --gres=gpu:{gpu_resource} \
    --wrap="{cmd_str}" \
    -o "slurm_logs/slurm-%j.out"'

    subprocess.run(slurm_cmd, shell=True)


if __name__ == "__main__":
    main()
