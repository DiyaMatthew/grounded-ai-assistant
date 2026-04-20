"""
backend/rag.py — Retrieval-Augmented Generation core.

Responsibilities:
  1. Load corpus.json built by data/build_corpus.py
  2. Chunk & embed all documents at startup (one-time cost)
  3. Expose a VectorStore.search() method used by app.py
  4. Provide a rerank() helper for better result quality

Architecture:
  corpus.json → VectorStore.build()
             → sentence-transformers (all-MiniLM-L6-v2)
             → numpy dot-product similarity search
             → top-k chunks returned with metadata

Usage (from app.py):
  from backend.rag import store
  results = store.search("What is RLHF?", top_k=3)
"""

from __future__ import annotations

import json
import logging
import pathlib
import re
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from sentence_transformers import SentenceTransformer

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────

MODEL_NAME   = "all-MiniLM-L6-v2"   # 22 MB, fast, strong for semantic search
CORPUS_PATH  = "data/corpus.json"
CHUNK_SIZE   = 400                   # words per chunk
CHUNK_OVERLAP = 60                   # word overlap between consecutive chunks
DEFAULT_TOP_K = 3                    # chunks returned per query

# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class Chunk:
    """A single retrievable unit of text."""
    id:          str
    text:        str
    title:       str
    url:         str
    source:      str                         # "arxiv" | "uk_gov"
    published:   str        = ""
    authors:     list[str]  = field(default_factory=list)
    chunk_index: int        = 0


@dataclass
class SearchResult:
    """One ranked result returned to the caller."""
    chunk:    Chunk
    score:    float

    @property
    def source_label(self) -> str:
        """Human-readable source badge e.g. 'arXiv' or 'UK Gov'."""
        return "arXiv" if self.chunk.source == "arxiv" else "UK Gov"

    def to_dict(self) -> dict:
        return {
            "text":   self.chunk.text,
            "title":  self.chunk.title,
            "url":    self.chunk.url,
            "source": self.source_label,
            "score":  round(self.score, 4),
        }


# ── Text utilities ────────────────────────────────────────────────────────────

def _clean(text: str) -> str:
    """Normalise whitespace and remove common artefacts."""
    text = re.sub(r"\s+", " ", text)
    text = text.replace("\u00a0", " ")     # non-breaking spaces
    return text.strip()


def _chunk_text(
    text: str,
    size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    """
    Split text into overlapping word-level windows.

    Example with size=5, overlap=2:
      "a b c d e f g h" → ["a b c d e", "d e f g h"]
    """
    words = text.split()
    if not words:
        return []

    chunks = []
    i = 0
    while i < len(words):
        chunk_words = words[i : i + size]
        chunk = " ".join(chunk_words)
        if len(chunk) > 80:           # skip tiny tail fragments
            chunks.append(chunk)
        i += size - overlap

    return chunks


# ── VectorStore ───────────────────────────────────────────────────────────────

class VectorStore:
    """
    In-memory vector store backed by numpy.

    For a portfolio project this is ideal — zero infrastructure,
    instant startup, easy to understand. Swap for FAISS or Pinecone
    when you need >100k chunks.
    """

    def __init__(self, corpus_path: str = CORPUS_PATH):
        self.corpus_path = corpus_path
        self.chunks:     list[Chunk]    = []
        self.embeddings: Optional[np.ndarray] = None
        self._model:     Optional[SentenceTransformer] = None
        self._ready = False

    # ── Public API ─────────────────────────────────────────────────────────

    def build(self) -> None:
        """
        Load corpus → chunk → embed. Called once at app startup.
        Typical time: 30–90 s depending on corpus size and CPU.
        """
        log.info("Loading embedding model: %s", MODEL_NAME)
        self._model = SentenceTransformer(MODEL_NAME)

        log.info("Reading corpus: %s", self.corpus_path)
        raw_docs = self._load_corpus()

        log.info("Chunking %d documents …", len(raw_docs))
        self.chunks = self._build_chunks(raw_docs)
        log.info("Total chunks: %d", len(self.chunks))

        log.info("Embedding chunks (this takes ~30–60 s on first run) …")
        t0 = time.time()
        texts = [c.text for c in self.chunks]
        self.embeddings = self._model.encode(
            texts,
            batch_size=64,
            show_progress_bar=True,
            normalize_embeddings=True,   # unit vectors → dot product == cosine
        )
        log.info(
            "Embeddings built in %.1f s  shape=%s",
            time.time() - t0,
            self.embeddings.shape,
        )
        self._ready = True

    def search(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        min_score: float = 0.25,
    ) -> list[SearchResult]:
        """
        Retrieve the top_k most relevant chunks for a query.

        Args:
            query:     Natural language question or phrase.
            top_k:     Number of results to return.
            min_score: Discard chunks below this cosine similarity.
                       0.25 is a loose threshold — tighten to 0.4
                       if you see irrelevant results.

        Returns:
            List of SearchResult, sorted by score descending.
        """
        if not self._ready:
            raise RuntimeError("VectorStore not built. Call store.build() first.")

        if not query or not query.strip():
            return []

        # Embed the query (normalised so dot product == cosine similarity)
        q_emb = self._model.encode(
            [_clean(query)],
            normalize_embeddings=True,
        )                                           # shape: (1, dim)

        # Dot product against all chunk embeddings
        scores: np.ndarray = (self.embeddings @ q_emb.T).squeeze()  # (N,)

        # Get indices of top-k scores above threshold
        candidate_idx = np.where(scores >= min_score)[0]
        if len(candidate_idx) == 0:
            # Relax threshold and return best available
            candidate_idx = np.arange(len(self.chunks))

        top_idx = candidate_idx[
            np.argsort(scores[candidate_idx])[::-1][:top_k]
        ]

        results = [
            SearchResult(chunk=self.chunks[i], score=float(scores[i]))
            for i in top_idx
        ]

        # Deduplicate by title — don't return 3 chunks from the same paper
        results = self._deduplicate(results, top_k=top_k)

        return results

    def stats(self) -> dict:
        """Return a summary dict — useful for /health endpoint."""
        if not self._ready:
            return {"ready": False}
        sources = {}
        for c in self.chunks:
            sources[c.source] = sources.get(c.source, 0) + 1
        return {
            "ready":        True,
            "total_chunks": len(self.chunks),
            "by_source":    sources,
            "model":        MODEL_NAME,
        }

    # ── Private helpers ────────────────────────────────────────────────────

    def _load_corpus(self) -> list[dict]:
        path = pathlib.Path(self.corpus_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Corpus not found at '{self.corpus_path}'. "
                "Run data/build_corpus.py first."
            )
        docs = json.loads(path.read_text(encoding="utf-8"))
        log.info("Loaded %d raw documents", len(docs))
        return docs

    def _build_chunks(self, docs: list[dict]) -> list[Chunk]:
        """
        Convert raw corpus docs into Chunk objects.

        If the doc already has a short text (pre-chunked by the ingest
        scripts), use it as-is. Otherwise re-chunk here.
        """
        chunks = []
        for doc in docs:
            text = _clean(doc.get("text", ""))
            if not text:
                continue

            word_count = len(text.split())

            # Already a good-sized chunk → use directly
            if word_count <= CHUNK_SIZE + 50:
                chunks.append(self._make_chunk(doc, text, doc.get("chunk_index", 0)))

            # Long doc → re-chunk
            else:
                for i, sub in enumerate(_chunk_text(text)):
                    chunks.append(self._make_chunk(doc, sub, i))

        return chunks

    @staticmethod
    def _make_chunk(doc: dict, text: str, index: int) -> Chunk:
        return Chunk(
            id=          f"{doc.get('id', 'doc')}_{index}",
            text=        text,
            title=       doc.get("title", "Unknown"),
            url=         doc.get("url",   ""),
            source=      doc.get("source", "unknown"),
            published=   doc.get("published", ""),
            authors=     doc.get("authors",   []),
            chunk_index= index,
        )

    @staticmethod
    def _deduplicate(
        results: list[SearchResult],
        top_k: int,
    ) -> list[SearchResult]:
        """
        Ensure at most 1 chunk per document title.
        Keeps the highest-scoring chunk from each source document.
        """
        seen_titles: set[str] = set()
        deduped = []
        for r in results:
            title = r.chunk.title
            if title not in seen_titles:
                seen_titles.add(title)
                deduped.append(r)
            if len(deduped) >= top_k:
                break
        return deduped


# ── Format helpers for app.py ─────────────────────────────────────────────────

def format_context(results: list[SearchResult]) -> str:
    """
    Build the context string injected into the LLM prompt.

    Each chunk is labelled with its source so the LLM can
    attribute claims naturally in its answer.
    """
    parts = []
    for i, r in enumerate(results, 1):
        parts.append(
            f"[Source {i}: {r.chunk.title} ({r.source_label})]\n{r.chunk.text}"
        )
    return "\n\n---\n\n".join(parts)


def format_sources(results: list[SearchResult]) -> list[dict]:
    """
    Build the sources list sent to the frontend after the answer.
    Frontend renders these as citation links under each response.
    """
    return [r.to_dict() for r in results]


# ── Singleton ─────────────────────────────────────────────────────────────────
# Import this in app.py:  from backend.rag import store
# It is built once at startup via the lifespan handler in app.py.

store = VectorStore()
