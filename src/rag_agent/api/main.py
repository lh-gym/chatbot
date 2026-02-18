"""FastAPI entrypoint for ingest/query/trace endpoints."""

from __future__ import annotations

import os
from dataclasses import asdict
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from rag_agent.agent.fallback import DeterministicPlanner
from rag_agent.agent.planner import RagAgentPlanner
from rag_agent.agent.registry import ToolRegistry
from rag_agent.agent.tools import register_builtin_tools
from rag_agent.config import AgentConfig, ChunkingConfig, RetrievalConfig
from rag_agent.ingest.chunker import SemanticSlidingChunker
from rag_agent.ingest.embedder import HashingEmbedder
from rag_agent.ingest.parser import ParserRegistry
from rag_agent.ingest.pipeline import IngestPipeline
from rag_agent.obs.tracing import TraceStore
from rag_agent.retrieval.fusion import FusionLayer
from rag_agent.retrieval.retriever import DualRouteRetriever
from rag_agent.retrieval.vector_store import InMemoryVectorStore


def _create_llm() -> Any:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None

    from langchain_openai import ChatOpenAI

    return ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), temperature=0)


class IngestRequest(BaseModel):
    path: str
    doc_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class QueryRequest(BaseModel):
    question: str = Field(min_length=1)
    chat_history: list[Any] = Field(default_factory=list)


class SourceSearchRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=20)
    source: str | None = None


app = FastAPI(title="RAG Agent System", version="0.1.0")

_parser_registry = ParserRegistry()
_chunker = SemanticSlidingChunker(ChunkingConfig(min_tokens=500, max_tokens=1200, overlap_tokens=120))
_embedder = HashingEmbedder()
_vector_store = InMemoryVectorStore()
_ingest_pipeline = IngestPipeline(_parser_registry, _chunker, _embedder, _vector_store)

_fusion = FusionLayer(RetrievalConfig())
_retriever = DualRouteRetriever(_vector_store, _embedder, _fusion)
_registry = ToolRegistry()
register_builtin_tools(_registry, _retriever)

_trace_store = TraceStore()
_llm = _create_llm()
_planner: RagAgentPlanner | DeterministicPlanner = (
    RagAgentPlanner(
        llm=_llm,
        tool_registry=_registry,
        trace_store=_trace_store,
        config=AgentConfig(),
    )
    if _llm is not None
    else DeterministicPlanner(
        tool_registry=_registry,
        trace_store=_trace_store,
        config=AgentConfig(),
    )
)


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "llm_configured": _llm is not None,
        "planner_mode": "langchain" if _llm is not None else "deterministic",
        "trace_count": len(_trace_store.list_recent(limit=1000)),
    }


@app.post("/ingest")
def ingest(request: IngestRequest) -> dict[str, Any]:
    try:
        chunks = _ingest_pipeline.ingest_path(
            request.path,
            doc_id=request.doc_id,
            extra_metadata=request.metadata,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {
        "chunks_created": len(chunks),
        "chunk_ids": [chunk.chunk_id for chunk in chunks],
    }


@app.post("/query")
def query(request: QueryRequest) -> dict[str, Any]:
    try:
        return _planner.invoke(request.question, chat_history=request.chat_history)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/sources/search")
def source_search(request: SourceSearchRequest) -> dict[str, Any]:
    metadata_filter = {"source": request.source} if request.source else None
    hits = _retriever.retrieve(
        request.query,
        metadata_filter=metadata_filter,
        top_k=request.top_k,
    )
    return {
        "items": [
            {
                "chunk_id": hit.chunk.chunk_id,
                "doc_id": hit.chunk.doc_id,
                "score": hit.score,
                "text": hit.chunk.text,
                "metadata": hit.chunk.metadata,
            }
            for hit in hits
        ]
    }


@app.get("/traces")
def traces(limit: int = 20) -> dict[str, Any]:
    records = [asdict(record) for record in _trace_store.list_recent(limit=limit)]
    return {"items": records}


@app.get("/traces/{trace_id}")
def trace_detail(trace_id: str) -> dict[str, Any]:
    try:
        record = _trace_store.get(trace_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return asdict(record)


@app.get("/metrics")
def metrics() -> dict[str, Any]:
    return _trace_store.summary()
