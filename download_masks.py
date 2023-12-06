import os
import requests
from requests.auth import HTTPBasicAuth

# Add your authentication details here
username = 'registeredusers'
password = 'only'

# Directory containing the fragment folders
base_dir = 'data/fragments'

# Iterate through each folder in base_dir
for folder in os.listdir(base_dir):
    # Check if the folder name starts with 'fragment'
    if folder.startswith('fragment'):
        fragment_id = folder.replace('fragment', '')
        mask_path = os.path.join(base_dir, folder, 'mask.png')

        # Check if mask.png does not exist in the folder
        if not os.path.exists(mask_path):
            # Construct the URL to download the mask.png
            mask_id = fragment_id
            if mask_id.__contains__("_"):
                mask_id = mask_id.split("_")[0]
            url = f'http://dl.ash2txt.org/full-scrolls/Scroll1.volpkg/paths/{fragment_id}/{mask_id}_mask.png'
            print(url)

            # Make a GET request with basic authentication to download the file
            response = requests.get(url, auth=HTTPBasicAuth(username, password))

            if response.status_code == 200:
                # Write the content of the response to a file
                with open(mask_path, 'wb') as file:
                    file.write(response.content)
                print(f'Downloaded mask.png for fragment {fragment_id}')
            else:
                print(f'Failed to download mask.png for fragment {fragment_id}: HTTP {response.status_code}')
