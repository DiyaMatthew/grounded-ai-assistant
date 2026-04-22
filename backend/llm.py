"""
backend/llm.py — LLM answer generation via Together AI.

Together AI uses an OpenAI-compatible API so the client is identical
to the OpenAI SDK — just a different base_url and key.

Free tier: $25 credit on signup, no card needed.
Sign up: https://api.together.xyz

Usage:
    async for token in stream_answer(query, context, session_id):
        # send token to client via SSE
"""

from __future__ import annotations

import logging
import os
from collections import defaultdict
from typing import AsyncGenerator

from openai import AsyncOpenAI

log = logging.getLogger(__name__)

# ── Client (lazy init) ────────────────────────────────────────────────────────

_client: AsyncOpenAI | None = None

def _get_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        api_key = os.environ.get("TOGETHER_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "TOGETHER_API_KEY not set. "
                "Add it to your .env file locally and Railway Variables in production."
            )
        _client = AsyncOpenAI(
            api_key=api_key,
            base_url="https://api.together.xyz/v1",
        )
    return _client


# ── Session memory ────────────────────────────────────────────────────────────

_MAX_HISTORY = 6
_sessions: dict[str, list[dict]] = defaultdict(list)

def _get_history(session_id: str) -> list[dict]:
    return _sessions[session_id][-_MAX_HISTORY:]

def _append_history(session_id: str, role: str, content: str) -> None:
    _sessions[session_id].append({"role": role, "content": content})
    if len(_sessions[session_id]) > _MAX_HISTORY * 4:
        _sessions[session_id] = _sessions[session_id][-_MAX_HISTORY:]


# ── System prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert AI research assistant with deep knowledge \
of machine learning, AI policy, and UK technology regulation.

Your answers are:
- Grounded ONLY in the context provided below each question
- Clear and concise — 2 to 4 paragraphs maximum
- Honest: if the context does not contain enough information, say so clearly
- Never prefixed with "Based on my knowledge:" — just answer directly

When the context contains information from UK Government sources, mention the \
policy relevance where appropriate. When citing arXiv papers, you may reference \
the paper title naturally in your answer."""


# ── Main streaming function ───────────────────────────────────────────────────

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
        session_id: Used to maintain conversation history per user session.

    Yields:
        str tokens as they stream from Together AI.
    """
    client = _get_client()

    user_message = f"""Context from knowledge base:
{context}

---
Question: {query}"""

    _append_history(session_id, "user", user_message)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        *_get_history(session_id),
    ]

    full_reply = []

    try:
        stream = await client.chat.completions.create(
            model="meta-llama/Llama-3-70b-chat-hf",
            messages=messages,
            temperature=0.15,
            max_tokens=600,
            stream=True,
        )

        async for chunk in stream:
            token = chunk.choices[0].delta.content or ""
            if token:
                full_reply.append(token)
                yield token

    except Exception as e:
        log.exception("Together AI error for session %s", session_id)
        yield f"\n\n[Error generating answer: {e}]"
        return

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
