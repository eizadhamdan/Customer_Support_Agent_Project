# kb/ingest_docs.py

import os
import re
import json
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
# from utils.preprocessing import clean_text
from preprocessing import clean_text

BASE_DOC_URLS = [
    "https://docs.atlan.com/",
    "https://developer.atlan.com/"
]

OUTPUT_FILE = os.path.join("knowledge_base.json")


def scrape_links(base_url: str, max_links: int = 30) -> list:
    """
    Scrape a base documentation page and return a list of links.

    Args:
        base_url (str): Root URL of documentation site.
        max_links (int): Max number of links to collect (to keep it lightweight).

    Returns:
        List of full URLs.
    """
    try:
        resp = requests.get(base_url, timeout=10)
        resp.raise_for_status()
    except Exception as e:
        print(f"Failed to fetch {base_url}: {e}")
        return []

    soup = BeautifulSoup(resp.text, "html.parser")
    anchors = soup.find_all("a", href=True)

    links = []
    for a in anchors:
        href = a["href"]
        if href.startswith("#"):  # skip internal anchors
            continue
        full_url = urljoin(base_url, href)
        if base_url in full_url and full_url not in links:
            links.append(full_url)
        if len(links) >= max_links:
            break

    return links


def scrape_page(url: str) -> str:
    """
    Extract visible text from a given documentation page.

    Args:
        url (str): Page URL.

    Returns:
        Cleaned text content.
    """
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
    except Exception as e:
        print(f"Failed to fetch {url}: {e}")
        return ""

    soup = BeautifulSoup(resp.text, "html.parser")

    # Remove scripts and styles
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()

    text = soup.get_text(separator=" ")
    text = re.sub(r"\s+", " ", text).strip()

    return clean_text(text)


def chunk_text(text: str, max_words: int = 150) -> list:
    """
    Split text into chunks of ~max_words length for embeddings.

    Args:
        text (str): Input text
        max_words (int): Max words per chunk

    Returns:
        List of text chunks
    """
    words = text.split()
    chunks = []
    for i in range(0, len(words), max_words):
        chunk = " ".join(words[i: i + max_words])
        if len(chunk.split()) > 10:  # ignore very small chunks
            chunks.append(chunk)
    return chunks


def build_knowledge_base():
    """
    Scrape docs + dev hub, chunk into passages, and save to JSON.
    """
    kb_entries = []

    for base_url in BASE_DOC_URLS:
        print(f"Scraping links from {base_url}...")
        links = scrape_links(base_url)

        for link in links:
            print(f"Scraping {link}")
            text = scrape_page(link)
            if not text:
                continue

            chunks = chunk_text(text)
            for chunk in chunks:
                kb_entries.append({
                    "text": chunk,
                    "source": link
                })

    print(f"Total KB entries: {len(kb_entries)}")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(kb_entries, f, indent=2, ensure_ascii=False)

    print(f"Knowledge base saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    build_knowledge_base()
