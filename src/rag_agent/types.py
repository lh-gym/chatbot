"""Shared domain models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class ParsedDocument:
    """A parsed source document before chunking."""

    doc_id: str
    text: str
    metadata: dict[str, Any]


@dataclass(slots=True)
class DocumentChunk:
    """A chunked section of a source document."""

    chunk_id: str
    doc_id: str
    text: str
    token_count: int
    metadata: dict[str, Any]


@dataclass(slots=True)
class ScoredChunk:
    """A retrieval result with score and route metadata."""

    chunk: DocumentChunk
    score: float
    route: str
    rank: int = 0


@dataclass(slots=True)
class ToolTrace:
    """Trace record for an executed tool call."""

    name: str
    input_payload: dict[str, Any]
    output_preview: str
    latency_ms: float
