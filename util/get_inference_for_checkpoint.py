from constants import *
from get_inference import get_inference_folder


def get_inference_for_checkpoint(frag_ids, checkpoint_path, host):
    for frag_id in frag_ids:
        get_inference_folder(fragment_id=frag_id, full_model_run_dir=checkpoint_path, hostname=host)


if __name__ == '__main__':
    fragments = FRAGMENTS.values()
    checkpoint = LIVELY_MEADOW

    get_inference_for_checkpoint(frag_ids=fragments, checkpoint_path=checkpoint, host="vast")
