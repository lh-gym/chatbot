"""Fusion strategies for multi-route retrieval results."""

from __future__ import annotations

from abc import ABC, abstractmethod

from rag_agent.config import RetrievalConfig
from rag_agent.types import ScoredChunk


class Reranker(ABC):
    """Reranker interface used after score fusion."""

    @abstractmethod
    def rerank(self, query: str, candidates: list[ScoredChunk]) -> list[ScoredChunk]:
        """Return candidates in the final ranking order."""


class KeywordOverlapReranker(Reranker):
    """Lightweight reranker using query-document lexical overlap."""

    def rerank(self, query: str, candidates: list[ScoredChunk]) -> list[ScoredChunk]:
        query_terms = set(query.lower().split())
        rescored: list[ScoredChunk] = []
        for item in candidates:
            chunk_terms = set(item.chunk.text.lower().split())
            overlap = len(query_terms & chunk_terms) / max(1, len(query_terms))
            rescored.append(
                ScoredChunk(
                    chunk=item.chunk,
                    score=(item.score * 0.8) + (overlap * 0.2),
                    route=item.route,
                    rank=item.rank,
                )
            )
        return sorted(rescored, key=lambda x: x.score, reverse=True)


class FusionLayer:
    """Fuses route outputs with normalization + RRF + reranking."""

    def __init__(
        self,
        config: RetrievalConfig | None = None,
        reranker: Reranker | None = None,
    ) -> None:
        self.config = config or RetrievalConfig()
        self.reranker = reranker or KeywordOverlapReranker()

    def fuse(
        self,
        query: str,
        route_results: dict[str, list[ScoredChunk]],
        *,
        top_k: int | None = None,
    ) -> list[ScoredChunk]:
        """Fuse and rerank results from multiple retrieval routes.

        Fusion process:
        1. Normalize scores per route to comparable [0, 1] range.
        2. Compute RRF score across routes to favor consistently high-ranked docs.
        3. Combine normalized and RRF signals.
        4. Apply reranker for final semantic adjustment.
        """

        if not route_results:
            return []

        normalized_routes = {
            route: self._normalize_scores(results) for route, results in route_results.items()
        }
        rrf_scores = self._rrf_scores(normalized_routes)

        merged: dict[str, ScoredChunk] = {}
        for route_name, items in normalized_routes.items():
            for item in items:
                rrf_bonus = rrf_scores.get(item.chunk.chunk_id, 0.0)
                combined = (item.score * 0.6) + (rrf_bonus * 0.4)
                current = merged.get(item.chunk.chunk_id)
                if current is None or combined > current.score:
                    merged[item.chunk.chunk_id] = ScoredChunk(
                        chunk=item.chunk,
                        score=combined,
                        route=route_name,
                        rank=item.rank,
                    )

        reranked = self.reranker.rerank(query, list(merged.values()))
        limit = top_k or self.config.final_k
        return [
            ScoredChunk(chunk=item.chunk, score=item.score, route=item.route, rank=i + 1)
            for i, item in enumerate(reranked[:limit])
        ]

    @staticmethod
    def _normalize_scores(items: list[ScoredChunk]) -> list[ScoredChunk]:
        if not items:
            return []
        raw_scores = [item.score for item in items]
        high = max(raw_scores)
        low = min(raw_scores)
        if high == low:
            return [
                ScoredChunk(chunk=item.chunk, score=1.0, route=item.route, rank=i + 1)
                for i, item in enumerate(items)
            ]

        normalized: list[ScoredChunk] = []
        for i, item in enumerate(items):
            score = (item.score - low) / (high - low)
            normalized.append(
                ScoredChunk(chunk=item.chunk, score=score, route=item.route, rank=i + 1)
            )
        return normalized

    def _rrf_scores(self, route_results: dict[str, list[ScoredChunk]]) -> dict[str, float]:
        scores: dict[str, float] = {}
        for items in route_results.values():
            for rank, item in enumerate(items, start=1):
                chunk_id = item.chunk.chunk_id
                scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (
                    self.config.rrf_k + rank
                )
        return scores
