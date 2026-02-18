"""Dual-route retriever with recall-focused candidate expansion."""

from __future__ import annotations

from typing import Any

from rag_agent.config import RetrievalConfig
from rag_agent.ingest.embedder import Embedder
from rag_agent.retrieval.fusion import FusionLayer
from rag_agent.retrieval.vector_store import VectorStore
from rag_agent.types import ScoredChunk


class DualRouteRetriever:
    """Combines metadata-filtered and semantic retrieval routes.

    The implementation intentionally oversamples route candidates before fusion,
    which increases the chance that relevant chunks survive to final top-5,
    improving practical Recall@5.
    """

    def __init__(
        self,
        vector_store: VectorStore,
        embedder: Embedder,
        fusion_layer: FusionLayer,
        config: RetrievalConfig | None = None,
    ) -> None:
        self.vector_store = vector_store
        self.embedder = embedder
        self.fusion_layer = fusion_layer
        self.config = config or RetrievalConfig()

    def retrieve(
        self,
        query: str,
        *,
        metadata_filter: dict[str, Any] | None = None,
        top_k: int | None = None,
    ) -> list[ScoredChunk]:
        final_k = top_k or self.config.final_k

        # Oversample route candidates to improve final Recall@5 after fusion.
        semantic_k = max(self.config.semantic_k, final_k * 4)
        metadata_k = max(self.config.metadata_k, final_k * 4)

        query_embedding = self.embedder.embed_query(query)
        semantic_results = self.vector_store.semantic_search(
            query_embedding=query_embedding,
            k=semantic_k,
            metadata_filter=metadata_filter,
        )
        metadata_results = self.vector_store.metadata_search(
            query_text=query,
            k=metadata_k,
            metadata_filter=metadata_filter,
        )

        return self.fusion_layer.fuse(
            query=query,
            route_results={
                "semantic": semantic_results,
                "metadata": metadata_results,
            },
            top_k=final_k,
        )
