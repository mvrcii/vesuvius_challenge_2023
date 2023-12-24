import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utility.constants import JAZZBIGGER_FRAG_ID, GRIMHUGE_FRAG_ID, BLUEBIGGER_FRAG_ID, SKYBIGGER_FRAG_ID, DEVASBIGGER_FRAG_ID, \
    TRAILBREAKER_FRAG_ID, TITLE1_FRAG_ID, TITLE2_FRAG_ID
from get_inference import get_inference_folder

sys.path.append('../')

from utility import AlphaBetaMeta


def get_inference_for_checkpoint(frag_ids, checkpoint_path, host):
    for frag_id in frag_ids:
        get_inference_folder(fragment_id=frag_id, full_model_run_dir=checkpoint_path, hostname=host, single=True)


if __name__ == '__main__':
    meta = AlphaBetaMeta()

    fragments = [BLUEBIGGER_FRAG_ID, GRIMHUGE_FRAG_ID, SKYBIGGER_FRAG_ID, DEVASBIGGER_FRAG_ID, TRAILBREAKER_FRAG_ID,
                 JAZZBIGGER_FRAG_ID, TITLE1_FRAG_ID, TITLE2_FRAG_ID]
    checkpoint = meta.get_previous_model()

    get_inference_for_checkpoint(frag_ids=fragments, checkpoint_path=checkpoint, host="vast")
