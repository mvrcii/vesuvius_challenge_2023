import os

RAW = "raw"
CLEANED = "cleaned"
PROCESSED = "processed"
BINARIZED = "binarized"

SINGLE_LAYER_LABEL_DIR = os.path.join("data", "single_layer")
FOUR_LAYER_LABEL_DIR = os.path.join("data", "four_layer")
TWELVE_LAYER_LABEL_DIR = os.path.join("data", "twelve_layer")

FOUR_LAYER_HANDMADE_LABEL_DIR = os.path.join("data", "four_layer_handmade")


def get_label_dir(layer_count: int, manual=False):
    if layer_count == 1:
        return SINGLE_LAYER_LABEL_DIR
    elif layer_count == 4:
        if manual:
            return FOUR_LAYER_HANDMADE_LABEL_DIR
        else:
            return FOUR_LAYER_LABEL_DIR
    elif layer_count == 12:
        return TWELVE_LAYER_LABEL_DIR


def build_label_dir(layer_count: int, _type: str, manual=False):
    label_dir = get_label_dir(layer_count, manual)
    return os.path.join(label_dir, _type)


SINGLE_LAYER_LABEL_BINARIZED_DIR = build_label_dir(layer_count=1, _type=BINARIZED)
FOUR_LAYER_LABEL_BINARIZED_DIR = build_label_dir(layer_count=4, _type=BINARIZED)
TWELVE_LAYER_LABEL_BINARIZED_DIR = build_label_dir(layer_count=12, _type=BINARIZED)
FOUR_LAYER_LABEL_HANDMADE_DIR = build_label_dir(layer_count=4, _type=BINARIZED, manual=True)