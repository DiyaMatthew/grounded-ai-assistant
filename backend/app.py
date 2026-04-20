"""
backend/app.py — FastAPI application wiring rag.py + llm.py together.

Endpoints:
  GET /           → health check
  GET /search     → SSE streaming RAG answer + citations
  GET /health     → corpus stats (great for README badge)

Run:
  uvicorn backend.app:app --reload
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from backend.rag import format_context, format_sources, store
from backend.llm import stream_answer

from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

load_dotenv()
log = logging.getLogger(__name__)

# ── Lifespan: build vector store once at startup ──────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Building vector store …")
    # Run in thread pool so it doesn't block the event loop
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, store.build)
    log.info("Vector store ready. Starting server.")
    yield
    log.info("Shutting down.")


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="AI Semantic Search Assistant",
    version="2.0.0",
    lifespan=lifespan,
)
app.mount("/static", StaticFiles(directory="frontend"), name="static")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/ui")
async def serve_ui():
    return FileResponse("frontend/index.html")

@app.get("/")
async def root():
    return {"status": "ok", "message": "AI Semantic Search API v2"}


@app.get("/health")
async def health():
    """Returns corpus stats — link this in your README."""
    return store.stats()


@app.get("/search")
async def search(
    q:          str = Query(..., min_length=2, description="User query"),
    top_k:      int = Query(3,   ge=1, le=5,  description="Chunks to retrieve"),
    session_id: str = Query("default",        description="Session identifier"),
):
    """
    Main RAG endpoint. Returns a Server-Sent Events stream.

    Event types:
      {"type": "token",   "text": "..."}     ← streamed answer tokens
      {"type": "sources", "sources": [...]}  ← citation list (end of stream)
      {"type": "error",   "message": "..."}  ← on failure
    """

    async def event_stream():
        try:
            # 1. Retrieve relevant chunks
            results = store.search(q, top_k=top_k)

            if not results:
                yield _sse({"type": "token", "text": "I couldn't find relevant information in my knowledge base for that query."})
                yield "data: [DONE]\n\n"
                return

            context = format_context(results)
            sources = format_sources(results)

            # 2. Stream the LLM answer token by token
            async for token in stream_answer(q, context, session_id):
                yield _sse({"type": "token", "text": token})

            # 3. Send citations after the answer
            yield _sse({"type": "sources", "sources": sources})
            yield "data: [DONE]\n\n"

        except Exception as e:
            log.exception("Error in /search for query: %s", q)
            yield _sse({"type": "error", "message": str(e)})
            yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control":               "no-cache",
            "X-Accel-Buffering":           "no",   # disable nginx buffering
            "Access-Control-Allow-Origin": "*",
        },
    )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _sse(payload: dict) -> str:
    """Format a dict as a Server-Sent Event line."""
    return f"data: {json.dumps(payload)}\n\n"
