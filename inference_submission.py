import subprocess
import sys

from utility.checkpoints import CHECKPOINTS
from utility.fragments import RICHI_FRAG_ID, RICHI2_FRAG_ID, HOT_ROD_FRAG_ID


def main():
    # checkpoint_keys = ['wise-energy', 'olive-wind', 'curious-rain', 'desert-sea']
    checkpoint_keys = ['zesty-shape']

    # fragment_ids = [BLASTER_FRAG_ID, IRONHIDE_FRAG_ID, HOT_ROD_FRAG_ID, JETFIRE_FRAG_ID,
    #                 SKYGLORIOUS_FRAG_ID, THUNDERCRACKER_FRAG_ID, GRIMHUGE_FRAG_ID, JAZZBIGGER_FRAG_ID,
    #                 DEVASBIGGER_FRAG_ID, SUNSTREAKER_FRAG_ID, ULTRA_MAGNUS_FRAG_ID, BLUEBIGGER_FRAG_ID,
    #                 TRAILBIGGER_FRAG_ID]

    fragment_ids = [HOT_ROD_FRAG_ID]
    tta = False

    script = f"multilayer_approach/infer_layered_segmentation_padto16.py"
    if tta:
        script = f"multilayer_approach/infer_layered_segmentation_padto16_tta.py"

    for checkpoint_key in checkpoint_keys:
        checkpoint = CHECKPOINTS[checkpoint_key]

        for fragment_id in fragment_ids:
            print(f"Start Inference with {checkpoint} and {fragment_id}")
            command = [sys.executable,
                       script,
                       str(checkpoint), str(fragment_id),
                       '--stride', str(2)]

            if tta:
                command.append('--tta')

            try:
                result = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                print(result.stdout)
            except subprocess.CalledProcessError as e:
                print(f"Error occurred: {e.stderr}")


if __name__ == '__main__':
    main()
