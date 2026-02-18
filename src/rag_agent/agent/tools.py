"""Built-in tool implementations for the RAG agent."""

from __future__ import annotations

import base64
import re
import sqlite3
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from rag_agent.agent.registry import ToolRegistry, ToolSpec
from rag_agent.retrieval.retriever import DualRouteRetriever


class SearchToolInput(BaseModel):
    query: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=10)
    source: str | None = None


class SummarizeToolInput(BaseModel):
    text: str = Field(min_length=1)
    max_sentences: int = Field(default=3, ge=1, le=10)


class PolicyExtractToolInput(BaseModel):
    text: str = Field(min_length=1)


class OCRToolInput(BaseModel):
    image_path: str = Field(min_length=1)


class DbReadToolInput(BaseModel):
    key: str = Field(min_length=1)


class DbWriteToolInput(BaseModel):
    key: str = Field(min_length=1)
    value: str = Field(min_length=1)


def register_builtin_tools(
    registry: ToolRegistry,
    retriever: DualRouteRetriever,
    *,
    sqlite_path: str = "rag_agent.db",
    vision_llm: Any | None = None,
) -> None:
    """Register default tool set used by the planner.

    Tools:
    - `internal_search`: dual-route retrieval with chunk citations.
    - `summarize_text`: concise summarization utility.
    - `extract_policy`: rule/policy sentence extraction.
    - `ocr_vision`: OCR using OpenAI Vision when available.
    - `db_read` / `db_write`: local SQLite key-value persistence.
    """

    db_file = Path(sqlite_path)
    _ensure_kv_table(db_file)

    def _search(input_data: SearchToolInput) -> str:
        metadata_filter = {"source": input_data.source} if input_data.source else None
        hits = retriever.retrieve(
            input_data.query,
            metadata_filter=metadata_filter,
            top_k=input_data.top_k,
        )
        lines = []
        for hit in hits:
            snippet = _truncate(hit.chunk.text.replace("\n", " "), 220)
            lines.append(f"[{hit.chunk.chunk_id}] score={hit.score:.4f} {snippet}")
        if not lines:
            return "NO_RESULTS"
        return "\n".join(lines)

    def _summarize(input_data: SummarizeToolInput) -> str:
        sentences = [
            part.strip()
            for part in re.split(r"(?<=[.!?。！？])\s+", input_data.text)
            if part.strip()
        ]
        return " ".join(sentences[: input_data.max_sentences])

    def _extract_policy(input_data: PolicyExtractToolInput) -> str:
        lines = [line.strip() for line in input_data.text.splitlines() if line.strip()]
        policy_hits: list[str] = []
        keywords = ("must", "shall", "required", "禁止", "必须", "应当", "不得")
        for line in lines:
            if any(keyword in line.lower() for keyword in keywords):
                policy_hits.append(line)
        return "\n".join(policy_hits) if policy_hits else "NO_POLICY_FOUND"

    def _ocr(input_data: OCRToolInput) -> str:
        image_path = Path(input_data.image_path)
        if not image_path.exists():
            return f"IMAGE_NOT_FOUND: {input_data.image_path}"

        if vision_llm is None:
            return "OCR_SKIPPED: vision llm not configured"

        image_bytes = image_path.read_bytes()
        encoded = base64.b64encode(image_bytes).decode("utf-8")
        content = [
            {"type": "text", "text": "Extract all visible text faithfully."},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{encoded}"},
            },
        ]
        response = vision_llm.invoke([{ "role": "user", "content": content }])
        return str(getattr(response, "content", response))

    def _db_read(input_data: DbReadToolInput) -> str:
        with sqlite3.connect(db_file) as conn:
            cur = conn.execute("SELECT value FROM kv WHERE key = ?", (input_data.key,))
            row = cur.fetchone()
        return row[0] if row else "NOT_FOUND"

    def _db_write(input_data: DbWriteToolInput) -> str:
        with sqlite3.connect(db_file) as conn:
            conn.execute(
                "INSERT INTO kv(key, value) VALUES(?, ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (input_data.key, input_data.value),
            )
            conn.commit()
        return "OK"

    registry.register(
        ToolSpec(
            name="internal_search",
            description="Search indexed internal knowledge and return cited chunks.",
            args_schema=SearchToolInput,
            handler=_search,
            tags=["retrieval", "rag"],
        )
    )
    registry.register(
        ToolSpec(
            name="summarize_text",
            description="Summarize a text passage.",
            args_schema=SummarizeToolInput,
            handler=_summarize,
            tags=["nlp"],
        )
    )
    registry.register(
        ToolSpec(
            name="extract_policy",
            description="Extract policy/compliance sentences from text.",
            args_schema=PolicyExtractToolInput,
            handler=_extract_policy,
            tags=["policy"],
        )
    )
    registry.register(
        ToolSpec(
            name="ocr_vision",
            description="Run OCR on image content via vision model.",
            args_schema=OCRToolInput,
            handler=_ocr,
            tags=["vision", "ocr"],
        )
    )
    registry.register(
        ToolSpec(
            name="db_read",
            description="Read a value from local key-value store.",
            args_schema=DbReadToolInput,
            handler=_db_read,
            tags=["db"],
        )
    )
    registry.register(
        ToolSpec(
            name="db_write",
            description="Write a key-value pair into local store.",
            args_schema=DbWriteToolInput,
            handler=_db_write,
            tags=["db"],
        )
    )


def _truncate(text: str, max_length: int) -> str:
    if len(text) <= max_length:
        return text
    return text[: max_length - 3] + "..."


def _ensure_kv_table(db_path: Path) -> None:
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS kv (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
        )
        conn.commit()
