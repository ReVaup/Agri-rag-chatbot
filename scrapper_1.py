import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import re

BASE_URL = "https://oar.icrisat.org"
START_URL = "https://oar.icrisat.org/view/divisions/Genebank/"

HEADERS = {"User-Agent": "Mozilla/5.0"}


# ✅ Step 1: Get year links
def get_year_links():
    res = requests.get(START_URL, headers=HEADERS)
    soup = BeautifulSoup(res.text, "html.parser")

    year_links = []

    for a in soup.find_all("a", href=True):
        text = a.text.strip()

        if text.isdigit() and len(text) == 4:
            full_url = urljoin(START_URL, a["href"])
            year_links.append(full_url)

    return list(set(year_links))


# ✅ Step 2: Get article links
def get_article_links(year_url):
    res = requests.get(year_url, headers=HEADERS)

    # ✅ extract numeric article links
    matches = re.findall(r'href="(https?://[^"]*/\d+/|/\d+/)"', res.text)

    article_links = list(set(matches))

    # convert to full URLs
    article_links = [urljoin(year_url, link) for link in article_links]

    return article_links

# ✅ Step 3: Get PDF links
def get_pdf_link(article_url):
    try:
        res = requests.get(article_url, headers=HEADERS)
        soup = BeautifulSoup(res.text, "html.parser")

        for a in soup.find_all("a", href=True):
            href = a["href"]

            # ✅ exact condition based on your screenshot
            if href.endswith(".pdf"):
                return href  # already full URL

    except Exception as e:
        print(f"Error: {article_url} -> {e}")

    return None


# 🔥 MAIN PIPELINE
def main():
    pdf_links = []

    year_links = get_year_links()
    print(f"📅 Found {len(year_links)} year pages")

    for year in year_links:
        print(f"🔍 Processing year: {year}")

        articles = get_article_links(year)
        print(f"   Found {len(articles)} articles")

        for article in articles:
            pdf = get_pdf_link(article)
            if pdf:
                pdf_links.append(pdf)

    pdf_links = list(set(pdf_links))

    print(f"\n✅ Total PDFs found: {len(pdf_links)}")

    for pdf in pdf_links:
        print(pdf)


if __name__ == "__main__":
    main()