"""Deterministic fallback planner when external LLM is unavailable."""

from __future__ import annotations

import re
from typing import Any

from rag_agent.agent.registry import ToolRegistry
from rag_agent.config import AgentConfig
from rag_agent.obs.tracing import Timer, TraceStore, estimate_token_count
from rag_agent.types import ToolTrace


class DeterministicPlanner:
    """Planner that answers from retrieval evidence without LLM dependency.

    This implementation keeps the same response contract as `RagAgentPlanner`
    and is useful for local/offline environments where `OPENAI_API_KEY` is not
    configured. It always prioritizes citation-grounded output.
    """

    def __init__(
        self,
        *,
        tool_registry: ToolRegistry,
        trace_store: TraceStore,
        config: AgentConfig | None = None,
    ) -> None:
        self.tool_registry = tool_registry
        self.trace_store = trace_store
        self.config = config or AgentConfig()

    def invoke(
        self,
        question: str,
        *,
        chat_history: list[Any] | None = None,
    ) -> dict[str, Any]:
        del chat_history  # deterministic planner is stateless for now.
        observed_tools: list[ToolTrace] = []
        self.tool_registry.set_observer(observed_tools.append)
        try:
            with Timer() as timer:
                raw = self.tool_registry.execute(
                    "internal_search", {"query": question, "top_k": 5}
                )
        finally:
            self.tool_registry.set_observer(None)

        citations, snippets = _parse_search_output(raw)
        answer = _build_answer(snippets, citations)

        record = self.trace_store.create_record(
            question=question,
            answer=answer,
            citations=citations,
            source_snippets=snippets,
            tool_traces=observed_tools,
            input_tokens=estimate_token_count(question),
            output_tokens=estimate_token_count(answer),
            latency_ms=timer.elapsed_ms,
        )

        return {
            "answer": answer,
            "citations": citations,
            "trace_id": record.trace_id,
            "latency_ms": record.latency_ms,
            "latency_target_met": record.latency_ms
            <= (self.config.target_latency_seconds * 1000.0),
            "groundedness": record.groundedness,
            "groundedness_target_met": record.groundedness
            >= self.config.groundedness_target,
        }


def _parse_search_output(raw: str) -> tuple[list[str], list[str]]:
    citations: list[str] = []
    snippets: list[str] = []
    for line in raw.splitlines():
        match = re.match(r"^\[(?P<cid>[^\]]+)\]\s+score=[0-9.]+\s+(?P<body>.+)$", line)
        if not match:
            continue
        cid = match.group("cid").strip()
        body = match.group("body").strip()
        citations.append(cid)
        snippets.append(body)
    return citations, snippets


def _build_answer(snippets: list[str], citations: list[str]) -> str:
    if not snippets or not citations:
        return "无法在已索引文档中找到可验证证据。"

    lines: list[str] = []
    for idx, (snippet, citation) in enumerate(zip(snippets[:3], citations[:3], strict=True), start=1):
        lines.append(f"{idx}. {snippet} [{citation}]")
    return "\n".join(lines)

