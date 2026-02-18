"""End-to-end ingest pipeline: parse -> chunk -> embed -> upsert."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from rag_agent.ingest.chunker import SemanticSlidingChunker
from rag_agent.ingest.embedder import Embedder
from rag_agent.ingest.parser import ParserRegistry
from rag_agent.retrieval.vector_store import VectorStore
from rag_agent.types import DocumentChunk


class IngestPipeline:
    """Coordinates parser/chunker/embedder/vector store stages.

    This class isolates ingestion from query-time retrieval so indexing can be
    executed offline, in batch jobs, or during deployment warm-up.
    """

    def __init__(
        self,
        parser_registry: ParserRegistry,
        chunker: SemanticSlidingChunker,
        embedder: Embedder,
        vector_store: VectorStore,
    ) -> None:
        self._parser_registry = parser_registry
        self._chunker = chunker
        self._embedder = embedder
        self._vector_store = vector_store

    def ingest_path(
        self,
        path: str | Path,
        *,
        doc_id: str | None = None,
        extra_metadata: dict[str, Any] | None = None,
    ) -> list[DocumentChunk]:
        """Ingest a single source file and return created chunks."""

        parsed = self._parser_registry.parse_path(path, doc_id=doc_id)
        if extra_metadata:
            parsed.metadata.update(extra_metadata)

        chunks = self._chunker.chunk_document(parsed)
        embeddings = self._embedder.embed_documents([chunk.text for chunk in chunks])
        self._vector_store.upsert(chunks, embeddings)
        return chunks

    def ingest_many(self, paths: list[str | Path]) -> list[DocumentChunk]:
        """Ingest many paths and return flattened chunk list."""

        all_chunks: list[DocumentChunk] = []
        for path in paths:
            all_chunks.extend(self.ingest_path(path))
        return all_chunks
