FRAGMENTS = {
    "OPTIMUS": "20231024093300",
    "BUMBLEBEE": "20230702185752_superseded",
    "MEGATRON": "20230522181603",
    "STARSCREAM": "20230827161847",
    "SOUNDWAVE": "20230904135535",
    "IRONHIDE": "20230905134255",
    "RATCHET": "20230909121925",
    "JAZZ": "20231016151000",
    "WHEELJACK": "20231005123333_superseded",

    "ULTRA_MAGNUS": "20231106155351",
    "JETFIRE": "20231005123336",
    "BLASTER": "20230702185753",
    "GALVATRON": "20230929220925",
    "GRIMLOCK": "20231012184421",

    "SHOCKWAVE": "2"
}

OPTIMUS_FRAG_ID = FRAGMENTS["OPTIMUS"]
BUMBLEBEE_FRAG_ID = FRAGMENTS["BUMBLEBEE"]
MEGATRON_FRAG_ID = FRAGMENTS["MEGATRON"]
STARSCREAM_FRAG_ID = FRAGMENTS["STARSCREAM"]
SOUNDWAVE_FRAG_ID = FRAGMENTS["SOUNDWAVE"]
IRONHIDE_FRAG_ID = FRAGMENTS["IRONHIDE"]
RATCHET_FRAG_ID = FRAGMENTS["RATCHET"]
JAZZ_FRAG_ID = FRAGMENTS["JAZZ"]
ULTRA_MAGNUS_FRAG_ID = FRAGMENTS["ULTRA_MAGNUS"]
WHEELJACK_FRAG_ID = FRAGMENTS["WHEELJACK"]
JETFIRE_FRAG_ID = FRAGMENTS["JETFIRE"]
BLASTER_FRAG_ID = FRAGMENTS["BLASTER"]
GALVATRON_FRAG_ID = FRAGMENTS["GALVATRON"]
GRIMLOCK_FRAG_ID = FRAGMENTS["GRIMLOCK"]

SHOCKWAVE_FRAG_ID = FRAGMENTS["SHOCKWAVE"]


def get_frag_name_from_id(frag_id):
    for name, id in FRAGMENTS.items():
        if id == frag_id:
            return name
    return "Unknown Fragment"

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
# GRIMLOCK_FRAG_ID = "your_id_here_7"  # Replace 'your_id_here_7' with the actual ID
# ARCEE_FRAG_ID = "your_id_here_8"  # Replace 'your_id_here_8' with the actual ID
# DEVASTATOR_FRAG_ID = "your_id_here_10"  # Replace 'your_id_here_10' with the actual ID
