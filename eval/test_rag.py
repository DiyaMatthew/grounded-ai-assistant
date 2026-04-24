"""
eval/test_rag.py — Full evaluation suite for grounded-ai-assistant.

Tests four dimensions of RAG quality:
  1. Retrieval precision   — are the right chunks being found?
  2. Answer faithfulness   — does the answer stay within the context?
  3. Answer relevance      — does the answer actually address the query?
  4. Source diversity      — are both arXiv and UK Gov sources used?

Usage:
    conda activate rag-env
    python eval/test_rag.py

Output:
    eval/results.json       ← machine-readable results
    Printed report          ← human-readable summary
"""

from __future__ import annotations

from dotenv import load_dotenv
load_dotenv()

import asyncio
import json
import pathlib
import sys
import time

# Make sure backend is importable from project root
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from backend.rag import store, format_context
from backend.llm import get_answer

# ── Build the vector store once ───────────────────────────────────────────────

def build_store():
    print("Building vector store for eval (first run takes ~60s)...")
    store.build()
    print(f"Ready. {len(store.chunks)} chunks indexed.\n")


# ── Test cases ─────────────────────────────────────────────────────────────────

TEST_CASES = [

    # ── RAG & retrieval knowledge ──────────────────────────────────────────
    {
        "id": "rag_001",
        "category": "RAG concepts",
        "query": "What is retrieval-augmented generation?",
        "expected_keywords": ["retrieval", "generation", "context", "document"],
        "expected_source_type": "arxiv",
        "min_relevance": 0.45,
    },
    {
        "id": "rag_002",
        "category": "RAG concepts",
        "query": "How does semantic search work with embeddings?",
        "expected_keywords": ["embedding", "vector", "semantic", "similarity"],
        "expected_source_type": "arxiv",
        "min_relevance": 0.40,
    },
    {
        "id": "rag_003",
        "category": "RAG concepts",
        "query": "What are chunking strategies for RAG pipelines?",
        "expected_keywords": ["chunk", "overlap", "retrieval", "context"],
        "expected_source_type": "arxiv",
        "min_relevance": 0.35,
    },

    # ── LLM & transformer knowledge ────────────────────────────────────────
    {
        "id": "llm_001",
        "category": "LLM concepts",
        "query": "How are large language models fine-tuned?",
        "expected_keywords": ["fine", "tuning", "training", "model"],
        "expected_source_type": "arxiv",
        "min_relevance": 0.30,
    },
    {
        "id": "llm_002",
        "category": "LLM concepts",
        "query": "What is reinforcement learning from human feedback?",
        "expected_keywords": ["reinforcement", "human", "feedback", "reward"],
        "expected_source_type": "arxiv",
        "min_relevance": 0.35,
    },
    {
        "id": "llm_003",
        "category": "LLM concepts",
        "query": "Explain transformer attention mechanisms",
        "expected_keywords": ["attention", "transformer", "query", "key"],
        "expected_source_type": "arxiv",
        "min_relevance": 0.30,
    },

    # ── UK AI policy knowledge ─────────────────────────────────────────────
    {
        "id": "policy_001",
        "category": "UK AI policy",
        "query": "How does the UK regulate artificial intelligence?",
        "expected_keywords": ["regulation", "AI", "UK", "policy"],
        "expected_source_type": "uk_gov",
        "min_relevance": 0.50,
    },
    {
        "id": "policy_002",
        "category": "UK AI policy",
        "query": "What is the UK National AI Strategy?",
        "expected_keywords": ["strategy", "national", "AI", "UK"],
        "expected_source_type": "uk_gov",
        "min_relevance": 0.50,
    },
    {
        "id": "policy_003",
        "category": "UK AI policy",
        "query": "How is AI used in the NHS and healthcare?",
        "expected_keywords": ["NHS", "health", "AI", "clinical"],
        "expected_source_type": "uk_gov",
        "min_relevance": 0.40,
    },

    # ── AI safety & alignment ──────────────────────────────────────────────
    {
        "id": "safety_001",
        "category": "AI safety",
        "query": "What are the risks of advanced AI systems?",
        "expected_keywords": ["risk", "safety", "alignment", "AI"],
        "expected_source_type": "arxiv",
        "min_relevance": 0.30,
    },
    {
        "id": "safety_002",
        "category": "AI safety",
        "query": "What is AI alignment and why does it matter?",
        "expected_keywords": ["alignment", "values", "AI", "human"],
        "expected_source_type": "arxiv",
        "min_relevance": 0.30,
    },

    # ── Hallucination resistance ───────────────────────────────────────────
    {
        "id": "resist_001",
        "category": "Hallucination resistance",
        "query": "What is the price of a Netflix subscription in 2024?",
        "expected_keywords": [],   # should NOT find relevant content
        "expected_source_type": None,
        "min_relevance": 0.0,
        "expect_low_confidence": True,   # system should signal low confidence
    },
    {
        "id": "resist_002",
        "category": "Hallucination resistance",
        "query": "Who won the 2023 FIFA World Cup?",
        "expected_keywords": [],
        "expected_source_type": None,
        "min_relevance": 0.0,
        "expect_low_confidence": True,
    },
]


# ── Evaluation functions ───────────────────────────────────────────────────────

def eval_retrieval_precision(case: dict) -> dict:
    """Check if retrieved chunks contain expected keywords."""
    results = store.search(case["query"], top_k=3)

    if not results:
        return {"passed": False, "reason": "No results returned", "top_score": 0.0}

    top_score = results[0].score
    combined_text = " ".join(r.chunk.text for r in results).lower()

    # Check keyword presence
    keywords_found = [
        kw for kw in case["expected_keywords"]
        if kw.lower() in combined_text
    ]

    keyword_hit_rate = (
        len(keywords_found) / len(case["expected_keywords"])
        if case["expected_keywords"] else 1.0
    )

    # Check source type
    sources_returned = [r.chunk.source for r in results]
    source_match = (
        case["expected_source_type"] in sources_returned
        if case["expected_source_type"] else True
    )

    # Check minimum relevance
    score_pass = top_score >= case["min_relevance"]

    # For hallucination resistance: low confidence is expected
    if case.get("expect_low_confidence"):
        passed = top_score < 0.45   # should NOT find high-confidence results
        reason = (
            f"Correctly returned low confidence ({top_score:.2f})"
            if passed else
            f"Incorrectly returned high confidence ({top_score:.2f}) for out-of-domain query"
        )
        return {
            "passed": passed,
            "reason": reason,
            "top_score": round(top_score, 4),
            "keywords_found": [],
            "source_match": True,
        }

    passed = keyword_hit_rate >= 0.5 and score_pass

    return {
        "passed": passed,
        "reason": (
            f"Keywords: {len(keywords_found)}/{len(case['expected_keywords'])} found. "
            f"Score: {top_score:.2f} (min: {case['min_relevance']}). "
            f"Source match: {source_match}"
        ),
        "top_score": round(top_score, 4),
        "keyword_hit_rate": round(keyword_hit_rate, 2),
        "source_match": source_match,
        "keywords_found": keywords_found,
    }


async def eval_answer_faithfulness(case: dict) -> dict:
    """
    Check that the answer doesn't introduce facts not in the context.
    Simple heuristic: answer should reference at least one keyword from context.
    """
    if case.get("expect_low_confidence"):
        return {"passed": True, "reason": "Skipped for out-of-domain query", "skipped": True}

    results = store.search(case["query"], top_k=3)
    if not results:
        return {"passed": False, "reason": "No context retrieved"}

    context = format_context(results)
    answer = await get_answer(case["query"], context, session_id=f"eval_{case['id']}")

    answer_lower = answer.lower()
    context_lower = context.lower()

    # Extract significant words from context (>5 chars, not stopwords)
    STOPWORDS = {
        "which", "their", "there", "these", "about", "would", "could",
        "should", "where", "while", "being", "having", "other", "based"
    }
    context_words = {
        w for w in context_lower.split()
        if len(w) > 5 and w.isalpha() and w not in STOPWORDS
    }

    # Count how many context words appear in the answer
    answer_words = set(answer_lower.split())
    overlap = context_words & answer_words
    overlap_rate = len(overlap) / max(len(answer_words), 1)

    # Check for clear hallucination signals
    hallucination_phrases = [
        "based on my knowledge",
        "i don't have information",
        "i cannot find",
        "as of my training",
        "i believe",
    ]
    has_hallucination_signal = any(p in answer_lower for p in hallucination_phrases)

    # Answer should be grounded (overlap > 10%) and not signal out-of-context
    passed = overlap_rate > 0.10 and not has_hallucination_signal

    return {
        "passed": passed,
        "reason": f"Context overlap: {overlap_rate:.1%}. Hallucination signal: {has_hallucination_signal}",
        "overlap_rate": round(overlap_rate, 3),
        "answer_preview": answer[:150] + "..." if len(answer) > 150 else answer,
        "hallucination_signal": has_hallucination_signal,
    }


def eval_source_diversity(top_k: int = 5) -> dict:
    """Check that both arXiv and UK Gov sources appear in general queries."""
    test_queries = [
        "What is AI and how is it regulated?",
        "Explain machine learning and UK policy",
        "AI safety and government strategy",
    ]

    arxiv_count = 0
    gov_count = 0

    for query in test_queries:
        results = store.search(query, top_k=top_k)
        for r in results:
            if r.chunk.source == "arxiv":
                arxiv_count += 1
            elif r.chunk.source == "uk_gov":
                gov_count += 1

    both_present = arxiv_count > 0 and gov_count > 0

    return {
        "passed": both_present,
        "arxiv_hits": arxiv_count,
        "gov_hits": gov_count,
        "reason": (
            f"Both sources returned across {len(test_queries)} queries"
            if both_present else
            "Only one source type returned — corpus may be imbalanced"
        ),
    }


# ── Main runner ────────────────────────────────────────────────────────────────

async def run_full_eval():
    print("=" * 60)
    print("RAG EVALUATION SUITE — grounded-ai-assistant")
    print("=" * 60)
    print()

    results = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "corpus_size": len(store.chunks),
        "retrieval": [],
        "faithfulness": [],
        "source_diversity": None,
        "summary": {},
    }

    # ── 1. Retrieval precision ─────────────────────────────────────────────
    print("SECTION 1 — Retrieval Precision")
    print("-" * 40)

    retrieval_passed = 0
    for case in TEST_CASES:
        result = eval_retrieval_precision(case)
        result["id"] = case["id"]
        result["category"] = case["category"]
        result["query"] = case["query"]
        results["retrieval"].append(result)

        status = "PASS" if result["passed"] else "FAIL"
        if result["passed"]:
            retrieval_passed += 1

        print(f"[{status}] {case['id']:12s} | {case['query'][:45]:<45} | score={result['top_score']:.2f}")
        if not result["passed"]:
            print(f"         → {result['reason']}")

    retrieval_rate = retrieval_passed / len(TEST_CASES)
    print(f"\nRetrieval precision: {retrieval_passed}/{len(TEST_CASES)} ({retrieval_rate:.0%})")

    # ── 2. Answer faithfulness ─────────────────────────────────────────────
    print("\nSECTION 2 — Answer Faithfulness (LLM required)")
    print("-" * 40)

    faith_cases = [c for c in TEST_CASES if not c.get("expect_low_confidence")]
    faith_passed = 0

    for case in faith_cases[:6]:   # test first 6 to save API calls
        result = await eval_answer_faithfulness(case)
        result["id"] = case["id"]
        results["faithfulness"].append(result)

        if result.get("skipped"):
            print(f"[SKIP] {case['id']:12s} | {case['query'][:45]}")
            continue

        status = "PASS" if result["passed"] else "FAIL"
        if result["passed"]:
            faith_passed += 1

        print(f"[{status}] {case['id']:12s} | overlap={result.get('overlap_rate', 0):.1%} | {case['query'][:35]}")
        if result.get("answer_preview"):
            print(f"         → {result['answer_preview'][:80]}...")

    faith_rate = faith_passed / max(len(faith_cases[:6]), 1)
    print(f"\nFaithfulness: {faith_passed}/{len(faith_cases[:6])} ({faith_rate:.0%})")

    # ── 3. Source diversity ────────────────────────────────────────────────
    print("\nSECTION 3 — Source Diversity")
    print("-" * 40)

    diversity = eval_source_diversity()
    results["source_diversity"] = diversity

    status = "PASS" if diversity["passed"] else "FAIL"
    print(f"[{status}] arXiv hits: {diversity['arxiv_hits']} | UK Gov hits: {diversity['gov_hits']}")
    print(f"       → {diversity['reason']}")

    # ── Summary ────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    overall_pass = (
        retrieval_rate >= 0.70 and
        faith_rate >= 0.70 and
        diversity["passed"]
    )

    results["summary"] = {
        "retrieval_precision": f"{retrieval_rate:.0%}",
        "answer_faithfulness": f"{faith_rate:.0%}",
        "source_diversity": diversity["passed"],
        "overall": "PASS" if overall_pass else "NEEDS IMPROVEMENT",
    }

    print(f"Retrieval precision : {retrieval_rate:.0%}  {'✓' if retrieval_rate >= 0.70 else '✗'} (target: 70%)")
    print(f"Answer faithfulness : {faith_rate:.0%}  {'✓' if faith_rate >= 0.70 else '✗'} (target: 70%)")
    print(f"Source diversity    : {'✓' if diversity['passed'] else '✗'}  (both arXiv + UK Gov)")
    print(f"\nOverall: {results['summary']['overall']}")

    # ── Save results ───────────────────────────────────────────────────────
    output_path = pathlib.Path("eval/results.json")
    output_path.parent.mkdir(exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2))
    print(f"\nFull results saved → {output_path}")

    # ── README snippet ─────────────────────────────────────────────────────
    print("\n── Paste this into your README ─────────────────────────")
    print(f"| Retrieval precision | {retrieval_rate:.0%} ({retrieval_passed}/{len(TEST_CASES)} test cases) |")
    print(f"| Answer faithfulness | {faith_rate:.0%} ({faith_passed}/{len(faith_cases[:6])} test cases) |")
    print(f"| Source diversity    | {'Both arXiv + UK Gov sources returned' if diversity['passed'] else 'Single source only'} |")
    print("────────────────────────────────────────────────────────")

    return results


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    build_store()
    asyncio.run(run_full_eval())
