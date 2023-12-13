import json

DEFAULT_START_CENTER_LAYER = 25
DEFAULT_END_CENTER_LAYER = 45
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
        self.center_layers = tuple(data.get('center_layers_start_indices', (DEFAULT_START_CENTER_LAYER, DEFAULT_END_CENTER_LAYER)))
        self.best_12_layers = tuple(data.get('best_12_inclusive_layers', None))

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
    def __init__(self, file_path="fragments.json"):
        if not hasattr(self, 'initialized'):  # This check ensures __init__ is only run once
            self.initialized = True
            self.fragments = Fragment.load_from_json(file_path) if file_path else []
            self.fragment_dict = {fragment.name: fragment for fragment in self.fragments}

            self.FRAGMENTS = {fragment.name: fragment.id for fragment in self.fragments}
            self.FRAGMENTS_ALPHA = ['JETFIRE', 'GRIMLARGE', 'THUNDERCRACKER', 'SUNSTREAKER', 'JAZZILLA']
            self.FRAGMENTS_BETA = ['BLASTER', 'HOT_ROD', 'ULTRA_MAGNUS', 'DEVASTATOR', 'SKYWARP', 'IRONHIDE']

    def get_center_layers(self, frag_id):
        for fragment in self.fragments:
            if fragment.id == frag_id:
                return fragment.center_layers
        return DEFAULT_START_CENTER_LAYER, DEFAULT_END_CENTER_LAYER

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
