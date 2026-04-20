"""
build_corpus.py — Merge arxiv + gov docs into a single unified corpus.

Run this AFTER both ingest scripts have completed:
    python data/ingest_arxiv.py
    python data/ingest_gov_docs.py
    python data/build_corpus.py

Output:
    data/corpus.json        ← unified corpus used by rag.py
    data/corpus_stats.json  ← summary stats for your README
"""

import json
import pathlib
import re

ARXIV_PATH   = "data/arxiv_corpus.json"
GOV_PATH     = "data/gov_corpus.json"
OUTPUT_PATH  = "data/corpus.json"
STATS_PATH   = "data/corpus_stats.json"

# ── Chunking (applied to arxiv docs, gov docs already chunked) ────────────────

def chunk_text(text: str, size: int = 400, overlap: int = 60) -> list[str]:
    words = text.split()
    chunks, i = [], 0
    while i < len(words):
        chunk = " ".join(words[i : i + size])
        if len(chunk) > 80:
            chunks.append(chunk)
        i += size - overlap
    return chunks


def clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


# ── Process arxiv ─────────────────────────────────────────────────────────────

def process_arxiv(path: str) -> list[dict]:
    raw = json.loads(pathlib.Path(path).read_text())
    docs = []
    for paper in raw:
        for i, chunk in enumerate(chunk_text(paper["text"])):
            docs.append({
                "id": f"arxiv_{paper['id'].split('/')[-1]}_{i}",
                "title": paper["title"],
                "url": paper["url"],
                "published": paper.get("published", ""),
                "authors": paper.get("authors", []),
                "text": clean(chunk),
                "source": "arxiv",
                "chunk_index": i,
            })
    return docs


# ── Process gov docs ──────────────────────────────────────────────────────────

def process_gov(path: str) -> list[dict]:
    # Already chunked by ingest_gov_docs.py — just load and clean
    raw = json.loads(pathlib.Path(path).read_text())
    for doc in raw:
        doc["text"] = clean(doc["text"])
    return raw


# ── Main ──────────────────────────────────────────────────────────────────────

def build():
    print("Loading sources...")

    arxiv_docs = process_arxiv(ARXIV_PATH) if pathlib.Path(ARXIV_PATH).exists() else []
    gov_docs   = process_gov(GOV_PATH)     if pathlib.Path(GOV_PATH).exists()   else []

    print(f"  arXiv chunks : {len(arxiv_docs)}")
    print(f"  Gov doc chunks: {len(gov_docs)}")

    corpus = arxiv_docs + gov_docs

    # Deduplicate by text fingerprint (first 120 chars)
    seen, unique = set(), []
    for doc in corpus:
        fp = doc["text"][:120]
        if fp not in seen:
            seen.add(fp)
            unique.append(doc)

    print(f"  After dedup  : {len(unique)} chunks")

    # Write corpus
    pathlib.Path(OUTPUT_PATH).write_text(
        json.dumps(unique, indent=2, ensure_ascii=False)
    )

    # Write stats (paste these into your README!)
    sources = {}
    for doc in unique:
        s = doc["source"]
        sources[s] = sources.get(s, 0) + 1

    avg_words = sum(len(d["text"].split()) for d in unique) // len(unique)

    stats = {
        "total_chunks": len(unique),
        "by_source": sources,
        "avg_chunk_words": avg_words,
        "total_words": sum(len(d["text"].split()) for d in unique),
    }

    pathlib.Path(STATS_PATH).write_text(json.dumps(stats, indent=2))

    print(f"\nCorpus built → {OUTPUT_PATH}")
    print(f"Stats saved  → {STATS_PATH}")
    print("\n── README snippet ──────────────────────────────────────────")
    print(f"| Corpus     | {len(unique)} chunks from {len(sources)} sources |")
    print(f"| arXiv      | {sources.get('arxiv', 0)} paper chunks             |")
    print(f"| UK Gov     | {sources.get('uk_gov', 0)} policy doc chunks        |")
    print(f"| Avg chunk  | {avg_words} words                          |")
    print("────────────────────────────────────────────────────────────")


if __name__ == "__main__":
    build()
