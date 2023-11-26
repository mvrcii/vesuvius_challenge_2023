import requests
from bs4 import BeautifulSoup

def list_directories(url, username, password):
    # Send a GET request to the URL with authentication
    response = requests.get(url, auth=(username, password))

    # Check if the request was successful
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        directories = []

        # Find all 'a' tags and their next siblings (which contains the date)
        for link in soup.find_all('a'):
            href = link.get('href')
            if href and href.endswith('/'):
                # Get the date from the text next to the link
                date_text = link.next_sibling
                if date_text:
                    date_text = date_text.strip()  # Clean up any extra whitespace
                directories.append((href, date_text))

        return directories
    else:
        return f"Failed to retrieve data, status code: {response.status_code}"

# URL of the directory listing
url = "http://dl.ash2txt.org/full-scrolls/Scroll1.volpkg/paths/"

# Replace with your actual username and password
username = 'registeredusers'
password = 'only'

# Get the list of directories
directories = list_directories(url, username, password)
for directory, date in directories:
    print(f"Directory: {directory}, Date: {date}")
