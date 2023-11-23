FRAGMENTS = {
    "OPTIMUS": "20231024093300",
    "BUMBLEBEE": "20230702185752",
    "MEGATRON": "20230522181603",
    "STARSCREAM": "20230827161847",
    "SOUNDWAVE": "20230904135535",
    "IRONHIDE": "20230905134255",
    "RATCHET": "20230909121925",
    "JAZZ": "20231016151000",
    "ULTRA_MAGNUS": "20231106155351",

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

SHOCKWAVE_FRAG_ID = FRAGMENTS["SHOCKWAVE"]


def get_frag_name_from_id(frag_id):
    for name, id in FRAGMENTS.items():
        if id == frag_id:
            return name
    return "Unknown Fragment"

# Additional Transformers character names for IDs
# ULTRA_MAGNUS_FRAG_ID = "your_id_here_1"  # Replace 'your_id_here_1' with the actual ID
# SHOCKWAVE_FRAG_ID = "your_id_here_2"  # Replace 'your_id_here_2' with the actual ID
# HOT_ROD_FRAG_ID = "your_id_here_3"  # Replace 'your_id_here_3' with the actual ID
# JETFIRE_FRAG_ID = "your_id_here_4"  # Replace 'your_id_here_4' with the actual ID
# WHEELJACK_FRAG_ID = "your_id_here_5"  # Replace 'your_id_here_5' with the actual ID
# BLASTER_FRAG_ID = "your_id_here_6"  # Replace 'your_id_here_6' with the actual ID
# GRIMLOCK_FRAG_ID = "your_id_here_7"  # Replace 'your_id_here_7' with the actual ID
# ARCEE_FRAG_ID = "your_id_here_8"  # Replace 'your_id_here_8' with the actual ID
# GALVATRON_FRAG_ID = "your_id_here_9"  # Replace 'your_id_here_9' with the actual ID
# DEVASTATOR_FRAG_ID = "your_id_here_10"  # Replace 'your_id_here_10' with the actual ID
