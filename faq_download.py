import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import time

BASE_DOMAIN = "https://www.browserstack.com"
ROOT_URL = "https://www.browserstack.com/support/faq/browserstack-ai"
ALLOWED_PREFIX = ROOT_URL

visited = set()
all_text = []

def scrape_text_from_page(url):
    print(f"🔍 Scraping: {url}")
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")

        # Collect text from headings, paragraphs, and list items
        page_content = []
        for tag in soup.find_all(['h1', 'h2', 'h3', 'p', 'li']):
            text = tag.get_text(strip=True)
            if text:
                page_content.append(text)

        return "\n".join(page_content), soup

    except Exception as e:
        print(f"❌ Failed to fetch {url}: {e}")
        return "", None

def crawl_recursive(url):
    if url in visited or not url.startswith(ALLOWED_PREFIX):
        return

    visited.add(url)
    page_text, soup = scrape_text_from_page(url)

    if page_text:
        all_text.append(f"\n=== URL: {url} ===\n" + page_text)

    if soup:
        for a_tag in soup.find_all("a", href=True):
            next_link = urljoin(BASE_DOMAIN, a_tag["href"])
            if next_link.startswith(ALLOWED_PREFIX) and next_link not in visited:
                time.sleep(0.5)  # polite delay
                crawl_recursive(next_link)

# Start the crawl
crawl_recursive(ROOT_URL)

# Save all text to file
output_file = "browserstack_ai_faq_all.txt"
with open(output_file, "w", encoding="utf-8") as f:
    f.write("\n\n".join(all_text))

print(f"\n✅ All FAQ content saved to: {output_file}")
