import requests
from bs4 import BeautifulSoup
import pandas as pd

def get_area_from_file(base_url, directory, username, password):
    # Construct the URL for the area_cm2.txt file in the directory
    file_url = f"{base_url}{directory}area_cm2.txt"

    # Send a GET request to get the file content
    response = requests.get(file_url, auth=(username, password))
    if response.status_code == 200:
        # Assuming the file content is directly in the response body as plain text
        return response.text.strip()
    else:
        return None


def list_directories_and_areas(url, username, password):
    response = requests.get(url, auth=(username, password))
    print("response")
    print(response)
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        directories = []
        dates = {}
        areas = {}

        for link in soup.find_all('a'):
            print("link")
            print(link)
            href = link.get('href')
            if href.endswith('/'):
                directories.append(href)
                # Fetch area from the area_cm2.txt file in this directory
                area = get_area_from_file(url, href, username, password)
                print(area)
                areas[href] = -1
                if area is not None:
                    areas[href] = area

                date_text = link.next_sibling
                if date_text:
                    date_text = date_text.strip()  # Clean up any extra whitespace
                dates[href] = date_text

        return areas, dates
    else:
        return f"Failed to retrieve data, status code: {response.status_code}"


# URL of the directory listing
url = "http://dl.ash2txt.org/full-scrolls/Scroll1.volpkg/paths/"

# Replace with your actual username and password
username = 'registeredusers'
password = 'only'

# Get the areas from the directories
areas, dates = list_directories_and_areas(url, username, password)
# Convert the dictionary to a DataFrame
df = pd.DataFrame({"ID": areas.keys(), "Area": areas.values(), "Date": dates.values()})

# Save the DataFrame to a CSV file
df.to_csv('areas.csv', index=False)
print(areas)
