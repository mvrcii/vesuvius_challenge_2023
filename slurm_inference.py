import argparse
import re
import subprocess
import time

from scripts.download_inference_from_host import dynamic_closest_matches, get_checkpoint_name
from utility.checkpoints import CHECKPOINTS
from utility.fragments import get_frag_name_from_id, FragmentHandler


def print_colored(message, color, _print=True):
    colors = {
        "blue": '\033[94m',
        "green": '\033[92m',
        "red": '\033[91m',
        "purple": '\033[95m',
        "end": '\033[0m',
    }
    if _print:
        print(f"{colors[color]}{message}{colors['end']}")
    else:
        return f"{colors[color]}{message}{colors['end']}"


def get_fragment_id(fragment_id_or_name, confidence):
    fragment_ids = FragmentHandler().get_ids()
    fragment_names = FragmentHandler().get_names()
    name_to_id = FragmentHandler().FRAGMENTS

    # Check if input is an ID or a Name and convert to ID if necessary
    if fragment_id_or_name in fragment_ids or fragment_id_or_name in name_to_id:
        fragment_id = name_to_id.get(fragment_id_or_name, fragment_id_or_name)
    else:
        id_suggestions = dynamic_closest_matches(fragment_id_or_name, fragment_ids)
        name_suggestions = dynamic_closest_matches(fragment_id_or_name, fragment_names, threshold=confidence)
        suggestions = list(set(id_suggestions + name_suggestions))

        if suggestions:
            if len(suggestions) > 1:
                print("Did you mean one of the following?")
                for idx, suggestion in enumerate(suggestions, 1):
                    print(f"{idx}. {suggestion}")

                choice = input("Enter the number of the correct option, or 'n' to cancel: ")
                if choice.isdigit() and 1 <= int(choice) <= len(suggestions):
                    selected_suggestion = suggestions[int(choice) - 1]
                    print(selected_suggestion)
                    fragment_id = name_to_id.get(selected_suggestion, selected_suggestion)
                else:
                    print("Invalid selection.")
                    exit()
            elif len(suggestions) == 1:
                fragment_id = name_to_id.get(suggestions[0])
            else:
                print("No valid suggestion found.")
                exit()
        else:
            print_colored(f"No close match found for: {fragment_id_or_name}", "red")
            exit()

    return fragment_id


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

    fragment_id = get_fragment_id(fragment_id_or_name=args.fragment_id, confidence=0.8)
    checkpoint_path = get_checkpoint_name(args.checkpoint_path, checkpoint_dict=CHECKPOINTS)

    tta_str = "_tta" if args.tta else ""

    node_name = "tenant-ac-nowak-h100-reserved-237-02" if args.node2 else "tenant-ac-nowak-h100-reserved-164-01"
    script_name = f"multilayer_approach/infer_layered_segmentation_padto16{tta_str}.py"
    print_colored(f"INFO:\tUsing {script_name}", "blue")

    command = ["python3", str(script_name), str(checkpoint_path), str(fragment_id),
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
        checkpoint_name = "-".join(checkpoint_path.split('-')[0:2])
        tta_str = " + TTA" if args.tta else ""
        stride_str = f"S{args.stride}"
        message = f"INFO:\t{get_frag_name_from_id(fragment_id)} {stride_str}{tta_str} ({checkpoint_name}) {job_id}"
        print_colored(message=message, color="purple")
        if not args.no_tail:
            delay_seconds = 5  # Adjust this value as needed
            print_colored(f"INFO:\tWaiting for {delay_seconds} seconds before tailing the log file...", color="blue")
            time.sleep(delay_seconds)

            tail_cmd = f"tail -f logs/slurm-{job_id}.out"
            subprocess.run(tail_cmd, shell=True)
    else:
        print_colored("ERROR:\tFailed to submit job to Slurm or parse job ID.", color="red")


if __name__ == "__main__":
    main()
