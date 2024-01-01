import subprocess
import sys

from utility.checkpoints import CHECKPOINTS
from utility.fragments import IRONHIDE_FRAG_ID, BLASTER_FRAG_ID, SKYGLORIOUS_FRAG_ID, \
    JAZZBIGGER_FRAG_ID, THUNDERCRACKER_FRAG_ID, HOT_ROD_FRAG_ID, JETFIRE_FRAG_ID, GRIMHUGE_FRAG_ID, SUNSTREAKER_FRAG_ID, \
    ULTRA_MAGNUS_FRAG_ID, TRAILBIGGER_FRAG_ID, DEVASBIGGER_FRAG_ID, BLUEBIGGER_FRAG_ID


def main():
    checkpoints = [CHECKPOINTS['wise-energy'],
                   CHECKPOINTS['olive-wind'],
                   CHECKPOINTS['curious-rain'],
                   CHECKPOINTS['desert-sea']]

    fragment_ids = [BLASTER_FRAG_ID, IRONHIDE_FRAG_ID, HOT_ROD_FRAG_ID, JETFIRE_FRAG_ID,
                    SKYGLORIOUS_FRAG_ID, THUNDERCRACKER_FRAG_ID, GRIMHUGE_FRAG_ID, JAZZBIGGER_FRAG_ID,
                    DEVASBIGGER_FRAG_ID, SUNSTREAKER_FRAG_ID, ULTRA_MAGNUS_FRAG_ID, BLUEBIGGER_FRAG_ID,
                    TRAILBIGGER_FRAG_ID]

    tta = True

    for checkpoint in checkpoints:
        for fragment_id in fragment_ids:
            command = [sys.executable,
                       'multilayer_approach/infer_layered_segmentation_padto16_tta.py',
                       str(checkpoint), str(fragment_id),
                       '--stride', str(2)]

            if tta:
                command.append('--tta')

            subprocess.run(command, check=True)


if __name__ == '__main__':
    main()
