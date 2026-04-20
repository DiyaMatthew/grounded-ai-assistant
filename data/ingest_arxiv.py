"""
ingest_arxiv.py — Fetch AI papers from arXiv and build a corpus.

Usage:
    python data/ingest_arxiv.py

Output:
    data/arxiv_corpus.json
"""

import arxiv
import json
import time
import pathlib
import re

# ── Config ────────────────────────────────────────────────────────────────────

QUERIES = [
    "large language models",
    "retrieval augmented generation",
    "transformer neural network",
    "prompt engineering",
    "AI alignment safety",
    "diffusion models image generation",
    "reinforcement learning from human feedback",
    "semantic search embeddings",
    "fine tuning language models",
    "multimodal AI vision language",
]

PAPERS_PER_QUERY = 30        # 30 × 10 queries = 300 papers
OUTPUT_PATH = "data/arxiv_corpus.json"

# ── Helpers ───────────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """Remove LaTeX artefacts and normalise whitespace."""
    text = re.sub(r"\$[^$]+\$", "", text)       # inline LaTeX
    text = re.sub(r"\\[a-zA-Z]+\{[^}]*\}", "", text)  # \cmd{...}
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def paper_to_doc(paper) -> dict:
    """Convert an arxiv.Result into a corpus document."""
    # Combine title + abstract for richer retrieval context
    body = f"{paper.title}. {paper.summary}"
    return {
        "id": paper.entry_id,
        "title": paper.title,
        "url": paper.entry_id,
        "published": paper.published.strftime("%Y-%m-%d"),
        "authors": [a.name for a in paper.authors[:3]],  # first 3 authors
        "categories": paper.categories,
        "text": clean_text(body),
        "source": "arxiv",
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def ingest():
    pathlib.Path("data").mkdir(exist_ok=True)

    seen_ids = set()
    docs = []

    for query in QUERIES:
        print(f"\nFetching: '{query}'")

        search = arxiv.Search(
            query=query,
            max_results=PAPERS_PER_QUERY,
            sort_by=arxiv.SortCriterion.Relevance,
        )

        count = 0
        for paper in search.results():
            if paper.entry_id in seen_ids:
                continue
            seen_ids.add(paper.entry_id)
            docs.append(paper_to_doc(paper))
            count += 1

        print(f"  → {count} new papers (total: {len(docs)})")
        time.sleep(1)   # be polite to the arxiv API

    pathlib.Path(OUTPUT_PATH).write_text(
        json.dumps(docs, indent=2, ensure_ascii=False)
    )
    print(f"\nDone. Saved {len(docs)} papers → {OUTPUT_PATH}")


if __name__ == "__main__":
    ingest()
