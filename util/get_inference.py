import subprocess

from constants import *


def get_inference(frag_ids, checkpoint_path, host):
    for frag_id in frag_ids:
        command = [
            'bash',
            "./util/get_inference.sh",
            frag_id,
            checkpoint_path,
            host
        ]
        subprocess.run(command)


if __name__ == '__main__':
    # fragments = FRAGMENTS.values()
    fragments = FRAGMENT_IDS = [STARSCREAM_FRAG_ID, MEGATRON_FRAG_ID, IRONHIDE_FRAG_ID, RATCHET_FRAG_ID, SOUNDWAVE_FRAG_ID,
                OPTIMUS_FRAG_ID, ULTRA_MAGNUS_FRAG_ID, JAZZ_FRAG_ID, JETFIRE_FRAG_ID, SUNSTREAKER_FRAG_ID,
                DEVASTATOR_FRAG_ID,

                GRIMLOCK_FRAG_ID, HOT_ROD_FRAG_ID]

    get_inference(frag_ids=fragments, checkpoint_path=LIVELY_MEADOW, host="vast")
