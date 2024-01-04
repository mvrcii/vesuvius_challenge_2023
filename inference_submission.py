from multilayer_approach.infer_layered_segmentation_padto16 import infer_layered
from multilayer_approach.infer_layered_segmentation_padto16_tta import infer_layered_with_tta
from utility.checkpoints import CHECKPOINTS
from utility.fragments import HOT_ROD_FRAG_ID, BLASTER_FRAG_ID, IRONHIDE_FRAG_ID, JETFIRE_FRAG_ID, SKYGLORIOUS_FRAG_ID, \
    THUNDERCRACKER_FRAG_ID, GRIMHUGE_FRAG_ID, JAZZBIGGER_FRAG_ID, DEVASBIGGER_FRAG_ID, SUNSTREAKER_FRAG_ID, \
    BLUEBIGGER_FRAG_ID, ULTRA_MAGNUS_FRAG_ID, TRAILBIGGER_FRAG_ID


def main():
    checkpoint_keys = ['olive-wind', 'wise-energy', 'curious-rain', 'desert-sea']

    fragment_ids = [BLASTER_FRAG_ID, IRONHIDE_FRAG_ID, HOT_ROD_FRAG_ID, JETFIRE_FRAG_ID,
                    SKYGLORIOUS_FRAG_ID, THUNDERCRACKER_FRAG_ID, GRIMHUGE_FRAG_ID, JAZZBIGGER_FRAG_ID,
                    DEVASBIGGER_FRAG_ID, SUNSTREAKER_FRAG_ID, ULTRA_MAGNUS_FRAG_ID, BLUEBIGGER_FRAG_ID,
                    TRAILBIGGER_FRAG_ID]

    tta = True
    batch_size = 4
    stride = 2

    for checkpoint_key in checkpoint_keys:
        checkpoint = CHECKPOINTS[checkpoint_key]

        for fragment_id in fragment_ids:
            print(f"Start Inference with {checkpoint} and {fragment_id}")

            if tta:
                infer_layered_with_tta(checkpoint=checkpoint, frag_id=fragment_id, stride=stride)
            else:
                infer_layered(checkpoint=checkpoint, frag_id=fragment_id, stride=stride, batch_size=batch_size)


if __name__ == '__main__':
    main()
