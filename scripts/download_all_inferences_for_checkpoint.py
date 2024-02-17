import argparse
import logging
import subprocess

from utility.fragments import FragmentHandler


def main():
    parser = argparse.ArgumentParser(description='Download all Fragment Inferences for a checkpoint')
    parser.add_argument('checkpoint_path', type=str, help='Path to the configuration file')
    args = parser.parse_args()

    checkpoint = args.checkpoint_path
    fragments = FragmentHandler().get_ids()

    for frag_id in fragments:
        command = ["python",
                   "scripts/download_inference_from_host.py",
                   frag_id,
                   checkpoint,
                   "vast",
                   "--force"]

        try:
            subprocess.run(command)
        except Exception as e:
            logging.error(f"Error during subprocess call for fragment {frag_id}: {e}")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    main()
