import json
import os

from constants import get_frag_name_from_id


def write_to_config(path, **kwargs):
    os.makedirs(path, exist_ok=True)
    path = os.path.join(path, 'config.json')

    with open(path, 'w') as file:
        json.dump(kwargs, file, indent=4)


def write_config(config, frag_id, channels):
    frag_name = '_'.join([get_frag_name_from_id(frag_id)]).upper()
    target_dir = os.path.join(config.dataset_target_dir, str(config.patch_size), frag_name)

    write_to_config(target_dir, frag_id=frag_id, channels=channels)

    return target_dir
