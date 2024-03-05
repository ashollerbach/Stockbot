from bs4 import BeautifulSoup
from urllib.request import urlopen

url = 'https://www.nasdaq.com/market-activity/stocks/pbf/historical'
soup = BeautifulSoup(urlopen(url))

close = soup.find_all("td", class_="historical-data__cell")

for i in close:
    print(i.text)