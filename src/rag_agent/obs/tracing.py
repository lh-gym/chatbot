"""Tracing, cost accounting, and groundedness evaluation."""

from __future__ import annotations

import re
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone

from rag_agent.types import ToolTrace

_TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", flags=re.UNICODE)


@dataclass(slots=True)
class TraceRecord:
    trace_id: str
    timestamp_utc: str
    question: str
    answer: str
    citations: list[str]
    source_snippets: list[str]
    tool_traces: list[ToolTrace]
    input_tokens: int
    output_tokens: int
    estimated_cost_usd: float
    latency_ms: float
    groundedness: float


@dataclass(slots=True)
class CostModel:
    """Simple token pricing model (USD per 1K tokens)."""

    input_per_1k: float = 0.005
    output_per_1k: float = 0.015

    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        return (input_tokens / 1000.0) * self.input_per_1k + (
            output_tokens / 1000.0
        ) * self.output_per_1k


class GroundednessEvaluator:
    """Computes attribution correctness from answer-source overlap.

    Metric definition used here:
    - Split answer into sentences.
    - Remove citation tags like `[doc-1-chunk-0001]`.
    - A sentence is considered grounded if at least one source snippet has token
      overlap ratio >= `min_overlap`.

    This is a pragmatic, deterministic proxy suitable for CI/contract tests.
    """

    def __init__(self, min_overlap: float = 0.35) -> None:
        self.min_overlap = min_overlap

    def score(self, answer: str, source_snippets: list[str]) -> float:
        sentences = [
            sentence.strip()
            for sentence in re.split(r"(?<=[.!?。！？])\s+", answer)
            if sentence.strip()
        ]
        if not sentences:
            return 1.0
        if not source_snippets:
            return 0.0

        source_token_sets = [set(self._normalize(source)) for source in source_snippets]
        grounded = 0

        for sentence in sentences:
            clean_sentence = re.sub(r"\[[^\]]+\]", "", sentence).strip()
            sentence_tokens = set(self._normalize(clean_sentence))
            if not sentence_tokens:
                grounded += 1
                continue

            if any(
                self._overlap(sentence_tokens, source_tokens) >= self.min_overlap
                for source_tokens in source_token_sets
            ):
                grounded += 1

        return grounded / len(sentences)

    @staticmethod
    def _normalize(text: str) -> list[str]:
        return [token.lower() for token in _TOKEN_PATTERN.findall(text)]

    @staticmethod
    def _overlap(a: set[str], b: set[str]) -> float:
        if not a or not b:
            return 0.0
        return len(a & b) / len(a)


class TraceStore:
    """In-memory trace storage for API-level observability."""

    def __init__(
        self,
        *,
        cost_model: CostModel | None = None,
        groundedness_evaluator: GroundednessEvaluator | None = None,
    ) -> None:
        self._records: dict[str, TraceRecord] = {}
        self._cost_model = cost_model or CostModel()
        self._groundedness = groundedness_evaluator or GroundednessEvaluator()

    def create_record(
        self,
        *,
        question: str,
        answer: str,
        citations: list[str],
        source_snippets: list[str],
        tool_traces: list[ToolTrace],
        input_tokens: int,
        output_tokens: int,
        latency_ms: float,
    ) -> TraceRecord:
        trace_id = str(uuid.uuid4())
        groundedness = self._groundedness.score(answer, source_snippets)
        record = TraceRecord(
            trace_id=trace_id,
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
            question=question,
            answer=answer,
            citations=citations,
            source_snippets=source_snippets,
            tool_traces=tool_traces,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            estimated_cost_usd=self._cost_model.estimate_cost(input_tokens, output_tokens),
            latency_ms=latency_ms,
            groundedness=groundedness,
        )
        self._records[trace_id] = record
        return record

    def get(self, trace_id: str) -> TraceRecord:
        record = self._records.get(trace_id)
        if record is None:
            raise KeyError(f"Trace not found: {trace_id}")
        return record

    def list_recent(self, limit: int = 20) -> list[TraceRecord]:
        return list(self._records.values())[-limit:]

    def summary(self) -> dict[str, float | int]:
        """Aggregate core observability metrics for dashboard display."""
        records = list(self._records.values())
        total = len(records)
        if total == 0:
            return {
                "total_requests": 0,
                "avg_latency_ms": 0.0,
                "p95_latency_ms": 0.0,
                "avg_groundedness": 0.0,
                "total_input_tokens": 0,
                "total_output_tokens": 0,
                "total_estimated_cost_usd": 0.0,
            }

        latencies = sorted(record.latency_ms for record in records)
        p95_index = max(0, int((len(latencies) * 0.95) - 1))
        total_in = sum(record.input_tokens for record in records)
        total_out = sum(record.output_tokens for record in records)
        total_cost = sum(record.estimated_cost_usd for record in records)
        avg_groundedness = sum(record.groundedness for record in records) / total
        avg_latency = sum(latencies) / total

        return {
            "total_requests": total,
            "avg_latency_ms": avg_latency,
            "p95_latency_ms": latencies[p95_index],
            "avg_groundedness": avg_groundedness,
            "total_input_tokens": total_in,
            "total_output_tokens": total_out,
            "total_estimated_cost_usd": total_cost,
        }


class Timer:
    """Simple context timer used by planner."""

    def __init__(self) -> None:
        self._start = 0.0
        self.elapsed_ms = 0.0

    def __enter__(self) -> "Timer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.elapsed_ms = (time.perf_counter() - self._start) * 1000.0


def estimate_token_count(text: str) -> int:
    return len(_TOKEN_PATTERN.findall(text))
