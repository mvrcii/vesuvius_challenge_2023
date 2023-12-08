FRAGMENTS = {
    # TRAIN SET ALPHA
    "ULTRA_MAGNUS": "20231106155351",
    "OPTIMUS": "20231024093300",
    # "BUMBLEBEE": "20230702185752_superseded",
    "MEGATRON": "20230522181603",
    "STARSCREAM": "20230827161847",
    "SOUNDWAVE": "20230904135535",
    "IRONHIDE": "20230905134255",
    "RATCHET": "20230909121925",
    # "JAZZ": "20231016151000",
    "JAZZILLA": "20231016151001",
    "DEVASTATOR": "20231022170900",
    "SUNSTREAKER": "20231031143852",

    # TRAIN SET BETA / CURRENTLY USED FOR INFERENCE
    "BLASTER": "20230702185753",
    "JETFIRE": "20231005123336",
    "HOT_ROD": "20230929220926",
    # "GRIMLOCK": "20231012184421",
    "GRIMLARGE": "20231012184422",
    "SKYWARP": "20231007101615",
    "THUNDERCRACKER": "20231012173610",

    "SIDESWIPE": "20230701020044",  # infer next!

    # FRAGMENTS
    "ARCEE": "PHerc1667"
}

# TRAIN SET ALPHA
ULTRA_MAGNUS_FRAG_ID = FRAGMENTS["ULTRA_MAGNUS"]
OPTIMUS_FRAG_ID = FRAGMENTS["OPTIMUS"]
# BUMBLEBEE_FRAG_ID = FRAGMENTS["BUMBLEBEE"]
MEGATRON_FRAG_ID = FRAGMENTS["MEGATRON"]
STARSCREAM_FRAG_ID = FRAGMENTS["STARSCREAM"]
SOUNDWAVE_FRAG_ID = FRAGMENTS["SOUNDWAVE"]
IRONHIDE_FRAG_ID = FRAGMENTS["IRONHIDE"]
RATCHET_FRAG_ID = FRAGMENTS["RATCHET"]

# TRAIN SET BETA / CURRENTLY USED FOR INFERENCE
# JAZZ_FRAG_ID = FRAGMENTS["JAZZ"]
JAZZILLA_FRAG_ID = FRAGMENTS["JAZZILLA"]
JETFIRE_FRAG_ID = FRAGMENTS["JETFIRE"]
BLASTER_FRAG_ID = FRAGMENTS["BLASTER"]
HOT_ROD_FRAG_ID = FRAGMENTS["HOT_ROD"]
# GRIMLOCK_FRAG_ID = FRAGMENTS["GRIMLOCK"]
GRIMLARGE_FRAG_ID = FRAGMENTS["GRIMLARGE"]
DEVASTATOR_FRAG_ID = FRAGMENTS["DEVASTATOR"]
SKYWARP_FRAG_ID = FRAGMENTS["SKYWARP"]
THUNDERCRACKER_FRAG_ID = FRAGMENTS["THUNDERCRACKER"]
SUNSTREAKER_FRAG_ID = FRAGMENTS["SUNSTREAKER"]
SIDESWIPE_FRAG_ID = FRAGMENTS["SIDESWIPE"]

# FRAGMENTS
ARCEE_FRAG_ID = FRAGMENTS["ARCEE"]

FRAGMENTS_ALPHA = [GRIMLARGE_FRAG_ID, MEGATRON_FRAG_ID, IRONHIDE_FRAG_ID,
                   RATCHET_FRAG_ID, SOUNDWAVE_FRAG_ID, JETFIRE_FRAG_ID,
                   ULTRA_MAGNUS_FRAG_ID, DEVASTATOR_FRAG_ID, SIDESWIPE_FRAG_ID]

FRAGMENTS_BETA = [BLASTER_FRAG_ID, THUNDERCRACKER_FRAG_ID, HOT_ROD_FRAG_ID,
                  JAZZILLA_FRAG_ID, SUNSTREAKER_FRAG_ID, OPTIMUS_FRAG_ID,
                  STARSCREAM_FRAG_ID, SKYWARP_FRAG_ID, SIDESWIPE_FRAG_ID]

ROTATE = {
    # JAZZ_FRAG_ID: -1,
    JETFIRE_FRAG_ID: 0,
    SKYWARP_FRAG_ID: -1,
    DEVASTATOR_FRAG_ID: -1,
    BLASTER_FRAG_ID: -1,
    THUNDERCRACKER_FRAG_ID: -1,
}

FLIP = {
    JAZZILLA_FRAG_ID: None,
}

# CHECKPOINTS
CHECKPOINTS = {
    "chocolate-fog": "chocolate-fog-716-segformer-b2-231207-182217",
    "amber-plant": "amber-plant-717-segformer-b2-231208-003604",
    "revived-bee": "revived-bee-694-segformer-b2-231206-181839",
    "lively-meadow": "lively-meadow-695-segformer-b2-231206-230820",
    "stellar-violet": "stellar-violet-584-segformer-b2-231204-093958",
    "kind-donkey": "kind-donkey-583-segformer-b2-231204-001337",
    "elated-wind": "elated-wind-555-segformer-b2-231203-000033",
    "fine-wildflower": "fine-wildflower-497-segformer-b2-231128-164424",
    "solar-oath": "solar-oath-401-segformer-b2-231126-043455",
}

CHOCOLATE_FOG = CHECKPOINTS["chocolate-fog"]
AMBER_PLANT = CHECKPOINTS["amber-plant"]
REVIVED_BEE = CHECKPOINTS["revived-bee"]
LIVELY_MEADOW = CHECKPOINTS["lively-meadow"]  # trained on blaster only (auto generated labels)
STELLAR_VIOLET = CHECKPOINTS["stellar-violet"]
KIND_DONKEY = CHECKPOINTS["kind-donkey"]
ELATED_WIND = CHECKPOINTS["elated-wind"]
FINE_WILDFLOWER = CHECKPOINTS["fine-wildflower"]
SOLAR_OATH = CHECKPOINTS["solar-oath"]


def get_flip_value(frag_id):
    flip_val = FLIP.get(frag_id)
    if flip_val is not None:
        return flip_val
    return 0


def get_rotate_value(frag_id):
    rot_val = ROTATE.get(frag_id)
    if rot_val is not None:
        return rot_val
    return 0


def get_frag_name_from_id(frag_id):
    for name, id in FRAGMENTS.items():
        if id == frag_id:
            return name
    return "Unknown Fragment"


def get_ckpt_name_from_id(checkpoint_name):
    for name, checkpoint in CHECKPOINTS.items():
        if checkpoint == checkpoint_name:
            return name
    return "Unknown Checkpoint"


def get_frag_id_from_name(frag_name):
    for name, id in FRAGMENTS.items():
        if frag_name == name:
            return name
    return "Unknown Fragment"


def get_all_frag_infos():
    return [(name, FRAGMENTS[name]) for name in FRAGMENTS]


def get_all_frag_names():
    return FRAGMENTS.keys()


def get_all_frag_ids():
    return FRAGMENTS.values()

# Additional Transformers character names for IDs
# HOT_ROD_FRAG_ID = "your_id_here_3"  # Replace 'your_id_here_3' with the actual ID

# WHEELJACK
# GALVATRON
# SHOCKWAVE
# Sideswipe
# Prowl
# Hound
# Mirage
# Trailbreaker
# Wheelie
# Warpath
# Windcharger
# Bluestreak
