"""
ingest_gov_docs.py — Fetch UK Government AI policy documents.

Scrapes publicly available gov.uk HTML pages (no PDF parsing needed).
Falls back to requests + BeautifulSoup for clean text extraction.

Usage:
    pip install requests beautifulsoup4 lxml
    python data/ingest_gov_docs.py

Output:
    data/gov_corpus.json
"""

import requests
import json
import time
import pathlib
import re
from bs4 import BeautifulSoup

# ── UK Gov AI/Tech policy pages (all public, no auth needed) ─────────────────

GOV_SOURCES = [
    {
        "title": "UK National AI Strategy",
        "url": "https://www.gov.uk/government/publications/national-ai-strategy",
        "fetch_url": "https://www.gov.uk/government/publications/national-ai-strategy",
    },
    {
        "title": "A pro-innovation approach to AI regulation",
        "url": "https://www.gov.uk/government/publications/ai-regulation-a-pro-innovation-approach",
        "fetch_url": "https://www.gov.uk/government/publications/ai-regulation-a-pro-innovation-approach",
    },
    {
        "title": "AI Safety Summit outcomes",
        "url": "https://www.gov.uk/government/topical-events/ai-safety-summit-2023",
        "fetch_url": "https://www.gov.uk/government/topical-events/ai-safety-summit-2023",
    },
    {
        "title": "Responsible AI in government",
        "url": "https://www.gov.uk/guidance/responsible-use-of-artificial-intelligence-in-government",
        "fetch_url": "https://www.gov.uk/guidance/responsible-use-of-artificial-intelligence-in-government",
    },
    {
        "title": "Data Protection and AI",
        "url": "https://ico.org.uk/for-organisations/uk-gdpr-guidance-and-resources/artificial-intelligence/guidance-on-ai-and-data-protection/",
        "fetch_url": "https://ico.org.uk/for-organisations/uk-gdpr-guidance-and-resources/artificial-intelligence/guidance-on-ai-and-data-protection/",
    },
    {
        "title": "NHS AI Lab overview",
        "url": "https://transform.england.nhs.uk/ai-lab/",
        "fetch_url": "https://transform.england.nhs.uk/ai-lab/",
    },
    {
        "title": "Centre for Data Ethics and Innovation - AI assurance",
        "url": "https://www.gov.uk/government/organisations/centre-for-data-ethics-and-innovation",
        "fetch_url": "https://www.gov.uk/government/organisations/centre-for-data-ethics-and-innovation",
    },
    {
        "title": "AI and employment — CIPD guidance",
        "url": "https://www.cipd.org/uk/knowledge/guides/artificial-intelligence/",
        "fetch_url": "https://www.cipd.org/uk/knowledge/guides/artificial-intelligence/",
    },
]

# ── Helpers ───────────────────────────────────────────────────────────────────

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; RAG-Portfolio-Bot/1.0; "
        "+https://github.com/DiyaMatthew)"
    )
}

def fetch_page_text(url: str) -> str | None:
    """Fetch a URL and extract clean body text."""
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        resp.raise_for_status()
    except Exception as e:
        print(f"  ✗ Failed to fetch {url}: {e}")
        return None

    soup = BeautifulSoup(resp.text, "lxml")

    # Remove nav, header, footer, scripts, styles
    for tag in soup(["nav", "header", "footer", "script",
                     "style", "aside", "noscript"]):
        tag.decompose()

    # gov.uk main content lives in .govuk-main-wrapper or main
    main = (
        soup.find(class_="govuk-main-wrapper")
        or soup.find("main")
        or soup.find("article")
        or soup.body
    )

    if not main:
        return None

    text = main.get_text(separator=" ", strip=True)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def chunk_text(text: str, size: int = 500, overlap: int = 75) -> list[str]:
    """
    Split text into overlapping word-level chunks.
    500 words ≈ 650 tokens — ideal for llama3-70b context.
    """
    words = text.split()
    chunks, i = [], 0
    while i < len(words):
        chunk = " ".join(words[i : i + size])
        if len(chunk) > 100:   # skip tiny tail chunks
            chunks.append(chunk)
        i += size - overlap
    return chunks


# ── Main ──────────────────────────────────────────────────────────────────────

def ingest():
    pathlib.Path("data").mkdir(exist_ok=True)
    docs = []

    for source in GOV_SOURCES:
        print(f"\nFetching: {source['title']}")
        text = fetch_page_text(source["fetch_url"])

        if not text:
            print("  ✗ Skipped (no content)")
            continue

        chunks = chunk_text(text)
        print(f"  ✓ {len(chunks)} chunks extracted")

        for i, chunk in enumerate(chunks):
            docs.append({
                "id": f"gov_{len(docs)}",
                "title": source["title"],
                "url": source["url"],
                "chunk_index": i,
                "text": chunk,
                "source": "uk_gov",
            })

        time.sleep(2)   # polite delay between requests

    pathlib.Path("data/gov_corpus.json").write_text(
        json.dumps(docs, indent=2, ensure_ascii=False)
    )
    print(f"\nDone. Saved {len(docs)} chunks → data/gov_corpus.json")


if __name__ == "__main__":
    ingest()
