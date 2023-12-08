import argparse
import os
from datetime import datetime
from tqdm import tqdm
import numpy as np


def main(fragment__id, strategy):
    # Check for existing inference directories
    fragment_dir_path = os.path.join("inference", "results", f"fragment{fragment__id}")
    contained_dirs = os.listdir(fragment_dir_path)
    inference_dirs = [x for x in contained_dirs if os.path.isdir(x) and not x.startswith("ensemble")]

    # Check if enough directories exist
    if len(inference_dirs) < 2:
        print(f"{fragment_dir_path} only contains {len(inference_dirs)} inferences, ensemble not possible.")
        exit()

    # Determine name of output directory, e.g. ensemble1
    ensemble_dirs = [x for x in contained_dirs if os.path.isdir(x) and x.startswith("ensemble")]
    current_ensemble_ids = [int(dir_name.split("_")[-1]) for dir_name in ensemble_dirs]
    next_ensemble_id = 0
    if len(current_ensemble_ids) > 0:
        next_ensemble_id = max(current_ensemble_ids) + 1

    # Ensemble short info id
    ensemble_info_id = "".join([dir_name[0] for dir_name in inference_dirs]) + f"_{strategy}"

    # Output path
    out_path = os.path.join(fragment_dir_path, f"ensemble_{ensemble_info_id}_{next_ensemble_id}")
    os.makedirs(out_path)

    # Collect all .npy file names from each directory
    all_npy_files = {d: set([f for f in os.listdir(d) if f.endswith('.npy')]) for d in inference_dirs}

    # Find common files across all directories
    common_npy_files = set.intersection(*all_npy_files.values())

    # Find and print missing files in each directory
    for d in inference_dirs:
        missing_files = common_npy_files - all_npy_files[d]
        for file in missing_files:
            print(f"Skipping {file} as it is not contained in {d}")

    for npy_name in tqdm(common_npy_files):
        npy_files = []
        for inference_dir in inference_dirs:
            npy_path = os.path.join(fragment_dir_path, inference_dir, npy_name)
            npy_files.append(np.load(npy_path))

        # IMPLEMENT DIFFERENT STRATEGIES HERE
        result = np.mean(np.stack(npy_files), axis=0)
        np.save(os.path.join(out_path, npy_name), result)

    # Save config to output path, listing current time as well as the used inference directories
    config = f"Ensemble of {len(inference_dirs)} inferences, strategy: {strategy}\n"
    config += f"Used inferences:\n"
    for inference_dir in inference_dirs:
        config += f"{inference_dir}\n"
    config += f"Created at {datetime.now()}"
    with open(os.path.join(out_path, "config.txt"), "w") as f:
        f.write(config)

    print("Saved ensemble to", out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference on a fragment.")
    parser.add_argument("id", type=int, help="fragment id")
    parser.add_argument("-strategy", "--string", help="optional, specify strategy [avg] (default=avg)",
                        default="avg")

    args = parser.parse_args()
    print("Running ensemble for fragment", args.id, "with strategy", args.string, "...")
    main(args.id, args.string)
