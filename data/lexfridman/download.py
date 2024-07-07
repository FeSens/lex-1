import os
import requests
from bs4 import BeautifulSoup
import pandas as pd

# Function to scrape the website and collect the desired tags
def scrape_website(url):
    try:
        # Send a GET request to the website
        response = requests.get(url)
        response.raise_for_status()  # Check for request errors

        # Parse the HTML content
        soup = BeautifulSoup(response.text, 'html.parser')

        # Print the first 500 characters of the response to debug
        print(response.text[:500])

        # Find all <a> tags with href containing "Episode"
        episode_links = soup.find_all('a', href= lambda href: href and 'lexfridman.com' in href)

        # Extract and print the href attributes
        episode_urls = [link['href'] for link in episode_links if 'Episode' in link.text]
        return episode_urls

    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")
        return []

def download_mp3(url, download_folder):
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()

        file_name = url.split("/")[-1]
        file_path = os.path.join(download_folder, file_name)

        with open(file_path, 'wb') as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)

        return file_name
    except requests.exceptions.RequestException as e:
        print(f"Failed to download {url}: {e}")
        return None

def download_and_generate_metadata(episode_url, download_folder):
    try:
        response = requests.get(episode_url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        download_link = soup.find('a', class_='powerpress_link_d', title='Download')

        if download_link:
            mp3_url = download_link['href']
            file_name = download_mp3(mp3_url, download_folder)
            if file_name:
                return {
                    'episode_url': episode_url,
                    'mp3_url': mp3_url,
                    'file_name': file_name
                }
        else:
            print(f"No download link found for {episode_url}")
            return None
    except requests.exceptions.RequestException as e:
        print(f"Failed to process {episode_url}: {e}")
        return None

def main():
    website_url = 'https://lexfridman.com/podcast'
    download_folder = 'episodes'
    os.makedirs(download_folder, exist_ok=True)

    episode_urls = scrape_website(website_url)
    metadata = []

    for episode_url in episode_urls:
        if "https://lexfridman.com/sam-altman-2" not in episode_url:
            continue

        print(f"Processing {episode_url}...")
        episode_data = download_and_generate_metadata(episode_url, download_folder)
        if episode_data:
            metadata.append(episode_data)

    if metadata:
        df = pd.DataFrame(metadata)
        df.to_csv(os.path.join(download_folder, 'metadata.csv'), index=False)
        print("Metadata CSV created successfully.")

if __name__ == "__main__":
    main()
