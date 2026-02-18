"""Vector store interfaces and concrete adapters."""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Any, Protocol, cast

from rag_agent.types import DocumentChunk, ScoredChunk


class VectorStore(Protocol):
    """Minimal vector store contract for retrieval."""

    def upsert(self, chunks: list[DocumentChunk], embeddings: list[list[float]]) -> None:
        """Insert or update chunk vectors."""

    def semantic_search(
        self,
        query_embedding: list[float],
        k: int,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[ScoredChunk]:
        """Search by semantic vector similarity."""

    def metadata_search(
        self,
        query_text: str,
        k: int,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[ScoredChunk]:
        """Search by metadata-constrained lexical matching."""


@dataclass(slots=True)
class _StoredVector:
    chunk: DocumentChunk
    embedding: list[float]


class InMemoryVectorStore:
    """Deterministic vector store used for tests and local prototyping."""

    def __init__(self) -> None:
        self._store: dict[str, _StoredVector] = {}

    def upsert(self, chunks: list[DocumentChunk], embeddings: list[list[float]]) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must have the same length")
        for chunk, embedding in zip(chunks, embeddings, strict=True):
            self._store[chunk.chunk_id] = _StoredVector(chunk=chunk, embedding=embedding)

    def semantic_search(
        self,
        query_embedding: list[float],
        k: int,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[ScoredChunk]:
        candidates = [
            rec
            for rec in self._store.values()
            if _metadata_match(rec.chunk.metadata, metadata_filter)
        ]
        ranked = sorted(
            (
                ScoredChunk(
                    chunk=record.chunk,
                    score=_cosine_similarity(query_embedding, record.embedding),
                    route="semantic",
                )
                for record in candidates
            ),
            key=lambda item: item.score,
            reverse=True,
        )
        return [
            ScoredChunk(chunk=item.chunk, score=item.score, route=item.route, rank=i + 1)
            for i, item in enumerate(ranked[:k])
        ]

    def metadata_search(
        self,
        query_text: str,
        k: int,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[ScoredChunk]:
        query_tokens = set(query_text.lower().split())
        candidates = [
            rec
            for rec in self._store.values()
            if _metadata_match(rec.chunk.metadata, metadata_filter)
        ]

        scored = []
        for rec in candidates:
            text_tokens = set(rec.chunk.text.lower().split())
            overlap = len(query_tokens & text_tokens)
            denom = max(1, len(query_tokens))
            score = overlap / denom
            scored.append(ScoredChunk(chunk=rec.chunk, score=score, route="metadata"))

        ranked = sorted(scored, key=lambda item: item.score, reverse=True)
        return [
            ScoredChunk(chunk=item.chunk, score=item.score, route=item.route, rank=i + 1)
            for i, item in enumerate(ranked[:k])
        ]


class FaissVectorStoreAdapter:
    """FAISS adapter via LangChain community integration.

    This adapter keeps the same retrieval contract as `InMemoryVectorStore` so
    it can be swapped in production with minimal code changes.
    """

    def __init__(self, embedder: Any) -> None:
        try:
            from langchain_community.vectorstores import FAISS
            from langchain_core.documents import Document
            from langchain_core.embeddings import Embeddings
        except Exception as exc:  # pragma: no cover - import path is environment-dependent
            raise RuntimeError(
                "FAISS dependencies are not available. Install langchain-community/faiss-cpu."
            ) from exc

        class _EmbeddingAdapter(Embeddings):
            def __init__(self, adapter_embedder: Any) -> None:
                self._embedder = adapter_embedder

            def embed_documents(self, texts: list[str]) -> list[list[float]]:
                return cast(list[list[float]], self._embedder.embed_documents(texts))

            def embed_query(self, text: str) -> list[float]:
                return cast(list[float], self._embedder.embed_query(text))

        self._faiss_cls = FAISS
        self._document_cls = Document
        self._embeddings = _EmbeddingAdapter(embedder)
        self._index: Any | None = None

    def upsert(self, chunks: list[DocumentChunk], embeddings: list[list[float]]) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must have the same length")
        docs = [
            self._document_cls(
                page_content=chunk.text,
                metadata={**chunk.metadata, "chunk_id": chunk.chunk_id, "doc_id": chunk.doc_id},
            )
            for chunk in chunks
        ]
        ids = [chunk.chunk_id for chunk in chunks]

        if self._index is None:
            self._index = self._faiss_cls.from_embeddings(
                text_embeddings=list(zip([chunk.text for chunk in chunks], embeddings, strict=True)),
                embedding=self._embeddings,
                metadatas=[doc.metadata for doc in docs],
                ids=ids,
            )
            return

        self._index.add_embeddings(
            text_embeddings=list(zip([chunk.text for chunk in chunks], embeddings, strict=True)),
            metadatas=[doc.metadata for doc in docs],
            ids=ids,
        )

    def semantic_search(
        self,
        query_embedding: list[float],
        k: int,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[ScoredChunk]:
        if self._index is None:
            return []
        docs_and_scores = self._index.similarity_search_with_relevance_scores_by_vector(
            embedding=query_embedding,
            k=k,
            filter=metadata_filter,
        )
        results: list[ScoredChunk] = []
        for rank, (doc, score) in enumerate(docs_and_scores, start=1):
            chunk = DocumentChunk(
                chunk_id=str(doc.metadata.get("chunk_id", f"faiss-{rank}")),
                doc_id=str(doc.metadata.get("doc_id", "unknown")),
                text=doc.page_content,
                token_count=max(1, len(doc.page_content.split())),
                metadata=dict(doc.metadata),
            )
            results.append(ScoredChunk(chunk=chunk, score=float(score), route="semantic", rank=rank))
        return results

    def metadata_search(
        self,
        query_text: str,
        k: int,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[ScoredChunk]:
        if self._index is None:
            return []
        docs_and_scores = self._index.similarity_search_with_relevance_scores(
            query=query_text,
            k=k,
            filter=metadata_filter,
        )
        results: list[ScoredChunk] = []
        for rank, (doc, score) in enumerate(docs_and_scores, start=1):
            chunk = DocumentChunk(
                chunk_id=str(doc.metadata.get("chunk_id", f"faiss-meta-{rank}")),
                doc_id=str(doc.metadata.get("doc_id", "unknown")),
                text=doc.page_content,
                token_count=max(1, len(doc.page_content.split())),
                metadata=dict(doc.metadata),
            )
            results.append(ScoredChunk(chunk=chunk, score=float(score), route="metadata", rank=rank))
        return results


def _metadata_match(
    metadata: dict[str, Any], metadata_filter: dict[str, Any] | None
) -> bool:
    if not metadata_filter:
        return True
    for key, value in metadata_filter.items():
        if metadata.get(key) != value:
            return False
    return True


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    numerator = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = sqrt(sum(x * x for x in a))
    norm_b = sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return numerator / (norm_a * norm_b)
