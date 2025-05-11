import time
import random
import logging
import csv
from newspaper import Article, build
from fake_useragent import UserAgent

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

news_sites = {
    "NPR (National Public Radio)": "https://www.npr.org/sections/news/",  # Use direct news section
    "PBS NewsHour": "https://www.pbs.org/newshour/",
    "BBC News": "https://www.bbc.com/news",
    "CBC News (Canadian Broadcasting Corporation)": "https://www.cbc.ca/news",
    "ABC News (Australian Broadcasting Corporation)": "https://www.abc.net.au/news/",
    "Deutsche Welle (DW)": "https://www.dw.com/en/",
    "France 24": "https://www.france24.com/en/",
    "Reuters": "https://www.reuters.com/",
    "ProPublica": "https://www.propublica.org/",
    "The Christian Science Monitor": "https://www.csmonitor.com/",
    "Al Jazeera English": "https://www.aljazeera.com/",
    "USA Today": "https://www.usatoday.com/",
    "Axios": "https://www.axios.com/",
    "Snopes": "https://www.snopes.com/",
    "PolitiFact": "https://www.politifact.com/",
    "FactCheck.org": "https://www.factcheck.org/",
    "Lead Stories": "https://leadstories.com/"
}

MAX_ARTICLES_PER_SOURCE = 500
OUTPUT_FILE = "scraped_articles.csv"
OPINION_KEYWORDS = ["opinion", "analysis", "editorial", "commentary", "perspective"]
MIN_LENGTH = 200

# Use a random user-agent
ua = UserAgent()
seen_urls = set()  # Avoid duplicates

try:
    # Open CSV file for writing
    with open(OUTPUT_FILE, "w", newline='', encoding="utf-8") as csvfile:
        fieldnames = ["title", "text", "url", "source"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        # Write header row
        writer.writeheader()
        
        total_articles = 0

        # Iterate through news sites
        for name, site_url in news_sites.items():
            logging.info(f"Scraping from: {name} - {site_url}")
            try:
                paper = build(site_url, memoize_articles=False)
            except Exception as e:
                logging.warning(f"Failed to build source {site_url}: {e}")
                continue

            count = 0
            for article in paper.articles:
                if count >= MAX_ARTICLES_PER_SOURCE:
                    break
                try:
                    if article.url in seen_urls:
                        continue

                    article.download()
                    time.sleep(random.uniform(1, 2))  # Throttle requests
                    article.parse()

                    # Skip opinion articles based on title keywords
                    if any(keyword in article.title.lower() for keyword in OPINION_KEYWORDS):
                        continue

                    if len(article.text) < MIN_LENGTH:  # Skip very short articles
                        continue

                    # Write the article data to CSV
                    writer.writerow({
                        "title": article.title,
                        "text": article.text,
                        "url": article.url,
                        "source": name
                    })
                    seen_urls.add(article.url)
                    count += 1
                    total_articles += 1
                    logging.info(f"Saved article: {article.title}")
                except Exception as e:
                    logging.warning(f"Error with article: {e}")

        logging.info(f"\n✅ Total articles collected: {total_articles}")
        logging.info(f"✅ Saved to {OUTPUT_FILE}")

except KeyboardInterrupt:
    logging.warning("⛔ Interrupted by user. Saving partial results...")
    # No need to save partial results here since the CSV is being written as we go
    logging.info(f"✅ Process interrupted, but articles were already being saved to {OUTPUT_FILE}")