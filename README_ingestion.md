# Data ingestion — quick reference

## Run order (do this once)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Fetch ~300 arXiv AI papers  (takes ~2 mins)
python data/ingest_arxiv.py

# 3. Fetch UK Gov AI policy pages  (takes ~1 min)
python data/ingest_gov_docs.py

# 4. Merge into unified corpus
python data/build_corpus.py
```

## What you get

| File | Contents |
|------|----------|
| `data/arxiv_corpus.json` | Raw arXiv paper metadata + abstracts |
| `data/gov_corpus.json`   | Chunked UK Gov policy page text |
| `data/corpus.json`       | **Unified corpus used by rag.py** |
| `data/corpus_stats.json` | Stats for README badges |

## Corpus breakdown (example output)

```
arXiv chunks :  847
Gov doc chunks:  94
After dedup  :  921 chunks
Avg chunk    :  387 words
```

## Add to your README

```markdown
## Corpus

| Source | Chunks |
|--------|--------|
| arXiv research papers | ~847 |
| UK Gov AI policy docs | ~94  |
| **Total**             | **~921** |

Built with `python data/build_corpus.py`. Re-run to refresh.
```

## Troubleshooting

**Gov pages returning 403?**
Some gov.uk pages block scrapers intermittently. Re-run after 10 mins,
or manually save the HTML and load it with `open("page.html")`.

**arXiv rate limiting?**
The script sleeps 1s between queries. If you hit a 429,
increase `time.sleep(3)` in `ingest_arxiv.py`.

**Corpus too small?**
Increase `PAPERS_PER_QUERY` in `ingest_arxiv.py` (max 300 per query on free tier).
Or add more topics to the `QUERIES` list.
