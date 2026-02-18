"""Embedding abstractions and deterministic baseline implementation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from hashlib import blake2b
from math import sqrt


class Embedder(ABC):
    """Embedder interface used by ingest and retrieval components."""

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed many documents."""

    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        """Embed one query."""


class HashingEmbedder(Embedder):
    """Deterministic sparse-like embedding without external model calls.

    This class is primarily used for local tests and deterministic integration
    tests. In production, replace it with OpenAI or other embedding providers.
    """

    def __init__(self, dimension: int = 256) -> None:
        self.dimension = dimension

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    def _embed(self, text: str) -> list[float]:
        vector = [0.0 for _ in range(self.dimension)]
        tokens = text.lower().split()
        if not tokens:
            return vector

        for token in tokens:
            digest = blake2b(token.encode("utf-8"), digest_size=8).digest()
            idx = int.from_bytes(digest[:4], "little") % self.dimension
            sign = -1.0 if digest[4] % 2 else 1.0
            vector[idx] += sign

        norm = sqrt(sum(value * value for value in vector))
        if norm == 0:
            return vector
        return [value / norm for value in vector]
