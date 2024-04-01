import argparse
import re
import subprocess
import time

from multilayer_approach.infer_layered_segmentation_padto16 import main as infer_layered
from scripts.download_inference_from_host import dynamic_closest_matches
from utility.checkpoints import CHECKPOINTS, get_checkpoint_name
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
    parser.add_argument('--bs', type=int, default=4, help='Batch Size (default: 4)')
    args = parser.parse_args()

    fragment_id = get_fragment_id(fragment_id_or_name=args.fragment_id, confidence=0.8)
    checkpoint_path, short_name = get_checkpoint_name(args.checkpoint_path, checkpoint_dict=CHECKPOINTS,short_names=True)

    script_name = f"multilayer_approach/infer_layered_segmentation_padto16.py"
    print_colored(f"INFO:\tUsing {script_name}", "blue")

    message = f"INFO:\t{fragment_id} {get_frag_name_from_id(fragment_id)} stride={str(args.stride)} ({short_name})"
    print_colored(message=message, color="purple")

    infer_layered(checkpoint=checkpoint_path,
                  fragment_id=fragment_id,
                  stride_factor=args.stride,
                  batch_size=args.bs)


if __name__ == "__main__":
    main()
