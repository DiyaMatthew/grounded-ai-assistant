from __future__ import annotations
import logging
import os
from collections import defaultdict
from typing import AsyncGenerator
from groq import AsyncGroq

log = logging.getLogger(__name__)

_client: AsyncGroq | None = None

def _get_client() -> AsyncGroq:
    global _client
    if _client is None:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise EnvironmentError("GROQ_API_KEY not set.")
        _client = AsyncGroq(api_key=api_key)
    return _client

_MAX_HISTORY = 6
_sessions: dict[str, list[dict]] = defaultdict(list)

def _get_history(session_id: str) -> list[dict]:
    return _sessions[session_id][-_MAX_HISTORY:]

def _append_history(session_id: str, role: str, content: str) -> None:
    _sessions[session_id].append({"role": role, "content": content})
    if len(_sessions[session_id]) > _MAX_HISTORY * 4:
        _sessions[session_id] = _sessions[session_id][-_MAX_HISTORY:]

SYSTEM_PROMPT = """You are an expert AI research assistant with deep knowledge \
of machine learning, AI policy, and UK technology regulation.
Answer ONLY using the context provided. Be clear and concise.
Never say 'Based on my knowledge' — just answer directly."""

async def stream_answer(
    query: str,
    context: str,
    session_id: str = "default",
) -> AsyncGenerator[str, None]:
    client = _get_client()
    user_message = f"Context:\n{context}\n\nQuestion: {query}"
    _append_history(session_id, "user", user_message)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        *_get_history(session_id),
    ]
    full_reply = []
    try:
        stream = await client.chat.completions.create(
            model="llama-3.1-8b-instant",
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
        log.exception("Groq error for session %s", session_id)
        yield f"\n\n[Error: {e}]"
        return
    _append_history(session_id, "assistant", "".join(full_reply))

async def get_answer(query: str, context: str, session_id: str = "eval") -> str:
    tokens = []
    async for token in stream_answer(query, context, session_id):
        tokens.append(token)
    return "".join(tokens)