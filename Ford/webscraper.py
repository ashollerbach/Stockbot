import requests
from bs4 import BeautifulSoup

# Function to scrape a webpage
def scrape_webpage(url):
    # Send a GET request to the URL
    response = requests.get(url)

    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        # Parse the HTML content
        soup = BeautifulSoup(response.content, 'html.parser')

        # Example: Extracting all <a> tags (links) from the webpage
        links = soup.find_all('a')
        
        # Example: Extracting text from all <p> tags (paragraphs) from the webpage
        paragraphs = soup.find_all('p')
        
        # Print out the extracted data
        print("Links found on the webpage:")
        for link in links:
            print(link.get('href'))
        
        print("\nParagraphs found on the webpage:")
        for paragraph in paragraphs:
            print(paragraph.get_text())
    else:
        print("Failed to retrieve webpage. Status code:", response.status_code)

# Example usage
url_to_scrape = 'https://www.nasdaq.com/market-activity/stocks/f/historical'
scrape_webpage(url_to_scrape)
