import os

import requests
from requests.auth import HTTPBasicAuth

from utility.fragments import SUPERSEDED_FRAGMENTS

GREEN = "\033[92m"
RED = "\033[91m"
BLUE = "\033[94m"
RESET = "\033[0m"


def download_masks():
    # Add your authentication details here
    username = 'registeredusers'
    password = 'only'

    base_dir = 'data/fragments'

    if not os.path.exists(base_dir):
        print(f"{RED}Base directory does not exist: {base_dir}{RESET}")
        return

    for folder in os.listdir(base_dir):
        if folder.startswith('fragment'):
            print(BLUE + f"Processing folder: {folder}" + RESET)
            fragment_id = folder.replace('fragment', '')
            mask_path = os.path.join(base_dir, folder, 'mask.png')

            # Check if mask.png does not exist in the folder
            if not os.path.exists(mask_path):
                mask_id = fragment_id.split("_")[0]

                url_frag_id = fragment_id
                if fragment_id in SUPERSEDED_FRAGMENTS:
                    print("Warning: Added suffix superseded to fragment id!")
                    url_frag_id += "_superseded"
                url = f'http://dl.ash2txt.org/full-scrolls/Scroll1.volpkg/paths/{url_frag_id}/{mask_id}_mask.png'

                if url_frag_id == "RICHI":
                    print("Warning: Unusual download url!")
                    url = f'http://dl.ash2txt.org/richi-uploads/ThaumatoAnakalyptor/scroll1/working_first_letter/thaumato_mask.png'

                try:
                    response = requests.get(url, auth=HTTPBasicAuth(username, password))

                    if response.status_code == 200:
                        with open(mask_path, 'wb') as file:
                            file.write(response.content)
                        print(GREEN + f'Downloaded mask.png for fragment {fragment_id}' + RESET)
                    else:
                        print(
                            RED + f'Failed to download mask.png for fragment {fragment_id}: HTTP {response.status_code}' + RESET)

                except requests.RequestException as e:
                    print(RED + f'Error during request: {e}' + RESET)


if __name__ == '__main__':
    download_masks()
