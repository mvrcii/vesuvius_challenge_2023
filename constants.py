import os

from fragment import FragmentHandler

fragments = FragmentHandler()

# TRAIN SET ALPHA
ULTRA_MAGNUS_FRAG_ID = fragments.get_id("ULTRA_MAGNUS")
IRONHIDE_FRAG_ID = fragments.get_id("IRONHIDE")
JAZZILLA_FRAG_ID = fragments.get_id("JAZZILLA")
JETFIRE_FRAG_ID = fragments.get_id("JETFIRE")
BLASTER_FRAG_ID = fragments.get_id("BLASTER")
HOT_ROD_FRAG_ID = fragments.get_id("HOT_ROD")
GRIMLARGE_FRAG_ID = fragments.get_id("GRIMLARGE")
DEVASTATOR_FRAG_ID = fragments.get_id("DEVASTATOR")
SKYWARP_FRAG_ID = fragments.get_id("SKYWARP")
THUNDERCRACKER_FRAG_ID = fragments.get_id("THUNDERCRACKER")
SUNSTREAKER_FRAG_ID = fragments.get_id("SUNSTREAKER")

FRAGMENTS_ALPHA = [JETFIRE_FRAG_ID, GRIMLARGE_FRAG_ID, THUNDERCRACKER_FRAG_ID,
                   SUNSTREAKER_FRAG_ID, JAZZILLA_FRAG_ID]

FRAGMENTS_BETA = [BLASTER_FRAG_ID, HOT_ROD_FRAG_ID, ULTRA_MAGNUS_FRAG_ID,
                  DEVASTATOR_FRAG_ID, SKYWARP_FRAG_ID, IRONHIDE_FRAG_ID]

# CHECKPOINTS
CHECKPOINTS = {
    "playful-firefly": "playful-firefly-737-segformer-b2-231209-143850",
    "chocolate-fog": "chocolate-fog-716-segformer-b2-231207-182217",
    "amber-plant": "amber-plant-717-segformer-b2-231208-003604",
    "revived-bee": "revived-bee-694-segformer-b2-231206-181839",
    "lively-meadow": "lively-meadow-695-segformer-b2-231206-230820",
    "stellar-violet": "stellar-violet-584-segformer-b2-231204-093958",
    "kind-donkey": "kind-donkey-583-segformer-b2-231204-001337",
    "elated-wind": "elated-wind-555-segformer-b2-231203-000033",
    "fine-wildflower": "fine-wildflower-497-segformer-b2-231128-164424",
    "solar-oath": "solar-oath-401-segformer-b2-231126-043455",
    "deft-yogurt": "deft-yogurt-738-segformer-b2-231209-215717",
    "upbeat-tree": "upbeat-tree-741-segformer-b2-231210-210131",
}

UPBEAT_TREE = CHECKPOINTS["upbeat-tree"]
DEFT_YOGURT = CHECKPOINTS["deft-yogurt"]
PLAYFUL_FIREFLY = CHECKPOINTS["playful-firefly"]
CHOCOLATE_FOG = CHECKPOINTS["chocolate-fog"]
AMBER_PLANT = CHECKPOINTS["amber-plant"]
REVIVED_BEE = CHECKPOINTS["revived-bee"]
LIVELY_MEADOW = CHECKPOINTS["lively-meadow"]  # trained on blaster only (auto generated labels)
STELLAR_VIOLET = CHECKPOINTS["stellar-violet"]
KIND_DONKEY = CHECKPOINTS["kind-donkey"]
ELATED_WIND = CHECKPOINTS["elated-wind"]
FINE_WILDFLOWER = CHECKPOINTS["fine-wildflower"]
SOLAR_OATH = CHECKPOINTS["solar-oath"]

# LABEL TYPES
HANDMADE_LABELS = "handmade"
GENERATED_LABELS = "model_generated"
LABEL_BASE_PATH = os.path.join("data", "base_label_files")
LABEL_BINARIZED_PATH = os.path.join("data", "base_label_binarized")

# ITERATION PHASE
ALPHA = "Alpha"
BETA = "Beta"

IT_2_MODEL = {
    # 0: DEFT_YOGURT,
    1: None,
}

# META INFORMATION

# The iteration of model to be trained currently. Increase this value as soon as you have a good
# performing model run, and you want to prepare the training for a follow-up model, including:
# while(True):
#       ...
#       - perform the model training        with iteration i
#       - perform the batch inference       with iteration i
#       - download the inference results    with iteration i
#       - copy the labels to base labels    with iteration i
#       - binarize the labels               with iteration i+1      >>> increment iteration before this step <<<
#       - create the dataset                with iteration i+1
#       - perform the model training        with iteration i+1
#       ...

ITERATION = 1
LABEL_TYPE = GENERATED_LABELS

def get_frag_name_from_id(frag_id):
    output = FragmentHandler().get_name(frag_id)
    return output


def get_ckpt_name_from_id(checkpoint_name):
    for name, checkpoint in CHECKPOINTS.items():
        if checkpoint == checkpoint_name:
            return name
    return "Unknown Checkpoint"


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
