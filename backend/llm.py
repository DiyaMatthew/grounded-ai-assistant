"""
backend/llm.py — LLM answer generation via Groq (llama3-70b).

Uses an async generator so app.py can stream tokens to the frontend
as they arrive, rather than waiting for the full response.

Usage:
    async for token in stream_answer(query, context, session_id):
        # send token to client via SSE
"""

from __future__ import annotations

import logging
import os
from collections import defaultdict
from typing import AsyncGenerator

from groq import AsyncGroq

log = logging.getLogger(__name__)

# ── Client (lazy init) ────────────────────────────────────────────────────────

_client: AsyncGroq | None = None

def _get_client() -> AsyncGroq:
    global _client
    if _client is None:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "GROQ_API_KEY not set. Add it to your .env file."
            )
        _client = AsyncGroq(api_key=api_key)
    return _client


# ── Session memory (simple in-process store) ──────────────────────────────────
# Stores last N turns per session_id.
# For production: replace with Redis.

_MAX_HISTORY = 6    # keep last 3 turns (user + assistant pairs)
_sessions: dict[str, list[dict]] = defaultdict(list)

def _get_history(session_id: str) -> list[dict]:
    return _sessions[session_id][-_MAX_HISTORY:]

def _append_history(session_id: str, role: str, content: str) -> None:
    _sessions[session_id].append({"role": role, "content": content})
    # Trim to avoid unbounded growth
    if len(_sessions[session_id]) > _MAX_HISTORY * 4:
        _sessions[session_id] = _sessions[session_id][-_MAX_HISTORY:]


# ── System prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert AI research assistant with deep knowledge \
of machine learning, AI policy, and UK technology regulation.

Your answers are:
- Grounded ONLY in the context provided below each question
- Clear and concise — 2 to 4 paragraphs maximum
- Honest: if the context doesn't contain enough information, say so clearly
- Never prefixed with "Based on my knowledge:" — just answer directly

When the context contains information from UK Government sources, mention the \
policy relevance where appropriate. When citing arXiv papers, you may reference \
the paper title naturally in your answer."""


# ── Main function ─────────────────────────────────────────────────────────────

async def stream_answer(
    query:      str,
    context:    str,
    session_id: str = "default",
) -> AsyncGenerator[str, None]:
    """
    Async generator that yields answer tokens one by one.

    Args:
        query:      The user's question.
        context:    Retrieved chunks formatted by rag.format_context().
        session_id: Used to maintain conversation history.

    Yields:
        str tokens as they stream from the Groq API.
    """
    client = _get_client()

    # Build the user message with injected context
    user_message = f"""Context from knowledge base:
{context}

---
Question: {query}"""

    # Add to history BEFORE the call so parallel requests don't race
    _append_history(session_id, "user", user_message)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        *_get_history(session_id),
    ]

    full_reply = []

    try:
        stream = await client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            temperature=0.15,       # low = more factual, less creative
            max_tokens=600,
            stream=True,
        )

        async for chunk in stream:
            token = chunk.choices[0].delta.content or ""
            if token:
                full_reply.append(token)
                yield token

    except Exception as e:
        log.exception("Groq API error for session %s", session_id)
        yield f"\n\n[Error generating answer: {e}]"
        return

    # Save assistant reply to history
    _append_history(session_id, "assistant", "".join(full_reply))


# ── Non-streaming convenience (for evals) ─────────────────────────────────────

async def get_answer(
    query:      str,
    context:    str,
    session_id: str = "eval",
) -> str:
    """Return the full answer as a string — useful in eval scripts."""
    tokens = []
    async for token in stream_answer(query, context, session_id):
        tokens.append(token)
    return "".join(tokens)
