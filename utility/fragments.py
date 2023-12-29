import json
import os

DEFAULT_START_CENTER_LAYER = 20
DEFAULT_END_CENTER_LAYER = 50
DEFAULT_BOOST_THRESHOLD = 0.00001
DEFAULT_ROTATION = 0
DEFAULT_FLIP = None


class Fragment:
    def __init__(self, data):
        self.id = data['id']
        self.name = data['name']
        self.rotation = data.get('rotation', DEFAULT_ROTATION)
        self.flip = data.get('flip', DEFAULT_FLIP)
        self.boost_threshold = data.get('boost_threshold', DEFAULT_BOOST_THRESHOLD)
        self.best_layers = tuple(
            data.get('best_layers_start_indices', (None, None)))
        self.best_12_layers = tuple(data.get('best_12_inclusive_layers', (None, None)))

    @classmethod
    def load_from_json(cls, file_path):
        with open(file_path, 'r') as file:
            data = json.load(file)
            return [cls(fragment) for fragment in data]

    def __repr__(self):
        return f"Fragment({self.__dict__})"


class SingletonMeta(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(SingletonMeta, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class FragmentHandler(metaclass=SingletonMeta):
    def __init__(self, file_path=os.path.join("utility", "fragments.json")):
        if not hasattr(self, 'initialized'):  # This check ensures __init__ is only run once
            self.initialized = True
            self.fragments = Fragment.load_from_json(file_path) if file_path else []
            self.fragment_dict = {fragment.name: fragment for fragment in self.fragments}

            self.FRAGMENTS = {fragment.name: fragment.id for fragment in self.fragments}

    def get_best_layers(self, frag_id):
        for fragment in self.fragments:
            if fragment.id == frag_id:
                return fragment.best_layers
        return None, None

    def get_best_12_layers(self, frag_id):
        for fragment in self.fragments:
            if fragment.id == frag_id:
                return fragment.best_12_layers
        return None, None

    def get_boost_threshold(self, frag_id):
        for fragment in self.fragments:
            if fragment.id == frag_id:
                return fragment.boost_threshold
        print("Warning: Default boost threshold used for fragment", self.get_name(frag_id))
        return DEFAULT_BOOST_THRESHOLD

    def get_flip(self, frag_id):
        for fragment in self.fragments:
            if fragment.id == frag_id:
                return fragment.flip
        return DEFAULT_FLIP

    # 0 -> flip vertical axis
    # 1 -> flip horizontal axis
    def get_rotation(self, frag_id):
        for fragment in self.fragments:
            if fragment.id == frag_id:
                return fragment.rotation
        return DEFAULT_ROTATION

    def get_name(self, frag_id):
        for name, id in self.FRAGMENTS.items():
            if str(id) == str(frag_id):
                return name
        return "Unknown Fragment"

    def get_id(self, frag_name):
        return self.FRAGMENTS.get(frag_name, "Unknown Fragment")

    def get_name_2_id(self):
        return [(name, self.FRAGMENTS[name]) for name in self.FRAGMENTS]

    def get_names(self):
        return list(self.FRAGMENTS.keys())

    def get_ids(self):
        return list(self.FRAGMENTS.values())

    def get_inference_fragments(self):
        return sorted(list(set(self.FRAGMENTS.values()) - set(FRAGMENTS_IGNORE)))


# TRAIN SET ALPHA
JETFIRE_FRAG_ID = FragmentHandler().get_id("JETFIRE")
ULTRA_MAGNUS_FRAG_ID = FragmentHandler().get_id("ULTRA_MAGNUS")
IRONHIDE_FRAG_ID = FragmentHandler().get_id("IRONHIDE")
BLASTER_FRAG_ID = FragmentHandler().get_id("BLASTER")
HOT_ROD_FRAG_ID = FragmentHandler().get_id("HOT_ROD")
THUNDERCRACKER_FRAG_ID = FragmentHandler().get_id("THUNDERCRACKER")
SUNSTREAKER_FRAG_ID = FragmentHandler().get_id("SUNSTREAKER")

JAZZILLA_FRAG_ID = FragmentHandler().get_id("JAZZILLA")
JAZZBIGGER_FRAG_ID = FragmentHandler().get_id("JAZZBIGGER")

GRIMLARGE_FRAG_ID = FragmentHandler().get_id("GRIMLARGE")
GRIMHUGE_FRAG_ID = FragmentHandler().get_id("GRIMHUGE")

DEVASTATOR_FRAG_ID = FragmentHandler().get_id("DEVASTATOR")
DEVASBIGGER_FRAG_ID = FragmentHandler().get_id("DEVASBIGGER")

BLUESTREAK_FRAG_ID = FragmentHandler().get_id("BLUESTREAK")
BLUEBIGGER_FRAG_ID = FragmentHandler().get_id("BLUEBIGGER")

SKYWARP_FRAG_ID = FragmentHandler().get_id("SKYWARP")
SKYBIGGER_FRAG_ID = FragmentHandler().get_id("SKYBIGGER")
SKYHUGE_FRAG_ID = FragmentHandler().get_id("SKYHUGE")
SKYLINE_FRAG_ID = FragmentHandler().get_id("SKYLINE")

TRAILBREAKER_FRAG_ID = FragmentHandler().get_id("TRAILBREAKER")
TRAILBIGGER_FRAG_ID = FragmentHandler().get_id("TRAILBIGGER")

TITLE1_FRAG_ID = FragmentHandler().get_id("TITLE1")
TITLE2_FRAG_ID = FragmentHandler().get_id("TITLE2")

FRAGMENTS_ALPHA = [JETFIRE_FRAG_ID, GRIMLARGE_FRAG_ID]

FRAGMENTS_BETA = [BLASTER_FRAG_ID, HOT_ROD_FRAG_ID, ULTRA_MAGNUS_FRAG_ID,
                  DEVASTATOR_FRAG_ID, SKYWARP_FRAG_ID, IRONHIDE_FRAG_ID]

FRAGMENTS_IGNORE = [TITLE1_FRAG_ID, TITLE2_FRAG_ID, SKYWARP_FRAG_ID, IRONHIDE_FRAG_ID]

SUPERSEDED_FRAGMENTS = [GRIMLARGE_FRAG_ID, JAZZILLA_FRAG_ID, DEVASTATOR_FRAG_ID,
                        SKYWARP_FRAG_ID, BLUESTREAK_FRAG_ID, SKYBIGGER_FRAG_ID,
                        TRAILBREAKER_FRAG_ID, SKYHUGE_FRAG_ID]


def get_frag_name_from_id(frag_id):
    return FragmentHandler().get_name(frag_id)
