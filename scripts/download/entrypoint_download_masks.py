import os

import requests
from requests.auth import HTTPBasicAuth


def download_masks():
    # ANSI Color Codes
    GREEN = "\033[92m"
    RED = "\033[91m"
    BLUE = "\033[94m"
    RESET = "\033[0m"

    # Add your authentication details here
    username = 'registeredusers'
    password = 'only'

    # Directory containing the fragment folders
    base_dir = 'data/fragments'

    print(os.listdir(base_dir))
    # Iterate through each folder in base_dir
    for folder in os.listdir(base_dir):
        # Check if the folder name starts with 'fragment'
        if folder.startswith('fragment'):
            print(BLUE + f"Processing folder: {folder}" + RESET)
            fragment_id = folder.replace('fragment', '')
            mask_path = os.path.join(base_dir, folder, 'mask.png')

            # Check if mask.png does not exist in the folder
            if not os.path.exists(mask_path):
                mask_id = fragment_id
                if mask_id.__contains__("_"):
                    mask_id = mask_id.split("_")[0]
                url = f'http://dl.ash2txt.org/full-scrolls/Scroll1.volpkg/paths/{fragment_id}/{mask_id}_mask.png'

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
