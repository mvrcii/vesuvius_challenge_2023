import os

CHECKPOINTS = {
    # "playful-firefly": "playful-firefly-737-segformer-b2-231209-143850",
    # "chocolate-fog": "chocolate-fog-716-segformer-b2-231207-182217",
    # "amber-plant": "amber-plant-717-segformer-b2-231208-003604",
    # "revived-bee": "revived-bee-694-segformer-b2-231206-181839",
    # "lively-meadow": "lively-meadow-695-segformer-b2-231206-230820",
    # "stellar-violet": "stellar-violet-584-segformer-b2-231204-093958",
    # "kind-donkey": "kind-donkey-583-segformer-b2-231204-001337",
    # "elated-wind": "elated-wind-555-segformer-b2-231203-000033",
    # "fine-wildflower": "fine-wildflower-497-segformer-b2-231128-164424",
    # "solar-oath": "solar-oath-401-segformer-b2-231126-043455",
    # "deft-yogurt": "deft-yogurt-738-segformer-b2-231209-215717",
    # "upbeat-tree": "upbeat-tree-741-segformer-b2-231210-210131",
    # "snowy-firebrand": "snowy-firebrand-929-segformer-b2-231223-010927",  # single layer, focal:dice 2:1, jetfire+grimlarge

    # UNETR
    # 128 BS
    "efficient-aardvark": "efficient-aardvark-1173-unetr-sf-b5-231229-082126",
    "desert-sea": "desert-sea-1183-unetr-sf-b5-231230-074546",
    "olive-wind": "olive-wind-1194-unetr-sf-b5-231231-064008",
    "curious-rain": "curious-rain-1193-unetr-sf-b5-231231-063741",
    "zesty-shape": "zesty-shape-1196-unetr-sf-b5-231231-135846",

    # 512 BS
    "playful-aardvark": "playful-aardvark-1152-unetr-sf-b5-231228-170431",
    "lively-night": "lively-night-1186-unetr-sf-b5-231230-083903",
    "driven-firefly": "driven-firefly-1177-unetr-sf-b5-231229-204010",
    "wise-energy": "wise-energy-1190-unetr-sf-b5-231231-055216",
}

CURIOUS_RAIN = CHECKPOINTS['curious-rain']
DESERT_SEA = CHECKPOINTS['desert-sea']
OLIVE_WIND = CHECKPOINTS['olive-wind']
WISE_ENERGY = CHECKPOINTS['wise-energy']

def get_checkpoint_name(user_input, checkpoint_dict, short_names=False):
    short_names = checkpoint_dict.keys()

    if len(user_input.split('-')) > 3:
        return user_input

    match = closest_match(user_input, short_names)
    if match:
        if short_names:
            return checkpoint_dict[match[0]], match[0]
        return checkpoint_dict[match[0]]
    else:
        print("Error with checkpoint")
        exit()


# UPBEAT_TREE = CHECKPOINTS["upbeat-tree"]
# DEFT_YOGURT = CHECKPOINTS["deft-yogurt"]
# PLAYFUL_FIREFLY = CHECKPOINTS["playful-firefly"]
# CHOCOLATE_FOG = CHECKPOINTS["chocolate-fog"]
# AMBER_PLANT = CHECKPOINTS["amber-plant"]
# REVIVED_BEE = CHECKPOINTS["revived-bee"]
# LIVELY_MEADOW = CHECKPOINTS["lively-meadow"]  # trained on blaster only (auto generated labels)
# STELLAR_VIOLET = CHECKPOINTS["stellar-violet"]
# KIND_DONKEY = CHECKPOINTS["kind-donkey"]
# ELATED_WIND = CHECKPOINTS["elated-wind"]
# FINE_WILDFLOWER = CHECKPOINTS["fine-wildflower"]
# SOLAR_OATH = CHECKPOINTS["solar-oath"]

# LABEL TYPES
HANDMADE_LABELS = "handmade"
GENERATED_LABELS = "model_generated"
LABEL_BASE_PATH = os.path.join("data", "base_label_files")
LABEL_BINARIZED_PATH = os.path.join("data", "base_label_binarized")
LABEL_BINARIZED_SINGLE_PATH = os.path.join("data", "base_label_binarized_single")

# ITERATION PHASE
ALPHA = "Alpha"
BETA = "Beta"

IT_2_MODEL = {
    # 0: UPBEAT_TREE,
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
