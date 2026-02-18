"""Semantic and sliding-window chunking implementation."""

from __future__ import annotations

import re
from dataclasses import dataclass
from hashlib import sha1

from rag_agent.config import ChunkingConfig
from rag_agent.types import DocumentChunk, ParsedDocument

_TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", flags=re.UNICODE)
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?。！？])\s+")


@dataclass(slots=True)
class _ChunkState:
    segments: list[str]
    tokens: list[str]


class SemanticSlidingChunker:
    """Builds context-preserving chunks with semantic boundaries and overlap.

    Design notes:
    1. Semantic segmentation first.
       The document is split by paragraph boundaries, then by sentence boundaries.
       A light semantic gate (token-set overlap) decides whether the next sentence
       belongs to the current segment or starts a new segment. This helps avoid
       splitting a coherent thought in the middle.

    2. Sliding-window packing second.
       Semantic segments are packed into chunks constrained by `[min_tokens,
       max_tokens]` where possible. When a chunk reaches capacity, it is finalized
       and the tail of the chunk (`overlap_tokens`) is copied into the next chunk.
       This overlap is the key mechanism that keeps cross-chunk context available
       for retrieval and downstream generation.

    3. Handling long segments.
       If one semantic segment itself exceeds `max_tokens`, it is sliced by a
       fixed-size sliding window (`window=max_tokens`, `stride=max_tokens-overlap`).
       This guarantees every produced chunk remains <= `max_tokens` while still
       preserving continuity between adjacent windows.

    The algorithm intentionally prioritizes paragraph integrity and semantic
    cohesion before applying strict token limits.
    """

    def __init__(self, config: ChunkingConfig | None = None) -> None:
        self.config = config or ChunkingConfig()
        if self.config.overlap_tokens >= self.config.max_tokens:
            raise ValueError("overlap_tokens must be less than max_tokens")

    def chunk_document(self, document: ParsedDocument) -> list[DocumentChunk]:
        """Chunk a parsed document into semantically coherent windows.

        Args:
            document: Parsed text and metadata from a parser.

        Returns:
            Ordered `DocumentChunk` objects with overlap and token counts.

        Implementation details:
        - The method first calls `_semantic_segments` to generate paragraph-aware
          semantic segments.
        - Segments are incrementally packed into a mutable chunk state until the
          next segment would exceed `max_tokens`.
        - On finalization, the last `overlap_tokens` of the completed chunk are
          copied into the next chunk's initial state.
        - For very long segments (> `max_tokens`), `_split_long_segment` is used.
          It produces windows with `stride=max_tokens-overlap_tokens`, ensuring:
          1) deterministic coverage of the full segment,
          2) bounded chunk size,
          3) continuity between consecutive windows.

        This two-pass strategy is more stable than naive fixed-length chunking
        because semantic grouping happens before hard token slicing.
        """

        segments = self._semantic_segments(document.text)
        output: list[_ChunkWithTokens] = []
        state = _ChunkState(segments=[], tokens=[])

        for segment in segments:
            segment_tokens = self._tokenize(segment)

            if len(segment_tokens) > self.config.max_tokens:
                if state.tokens:
                    output.append(self._finalize_chunk(document, state, len(output)))
                    state = _ChunkState(segments=[], tokens=[])
                output.extend(
                    self._split_long_segment(document, segment, len(output))
                )
                continue

            if len(state.tokens) + len(segment_tokens) <= self.config.max_tokens:
                state.segments.append(segment)
                state.tokens.extend(segment_tokens)
                continue

            if state.tokens:
                finalized = self._finalize_chunk(document, state, len(output))
                output.append(finalized)
                overlap_tokens = finalized.text_tokens[-self.config.overlap_tokens :]
                overlap_text = self._detokenize(overlap_tokens)
                new_segments = [overlap_text, segment] if overlap_text else [segment]
                state = _ChunkState(
                    segments=new_segments,
                    tokens=self._tokenize("\n\n".join(new_segments)),
                )
            else:
                state = _ChunkState(segments=[segment], tokens=segment_tokens)

        if state.tokens:
            output.append(self._finalize_chunk(document, state, len(output)))

        return [
            DocumentChunk(
                chunk_id=chunk.chunk_id,
                doc_id=chunk.doc_id,
                text=chunk.text,
                token_count=chunk.token_count,
                metadata=chunk.metadata,
            )
            for chunk in output
        ]

    def _semantic_segments(self, text: str) -> list[str]:
        segments: list[str] = []
        for paragraph in self._split_paragraphs(text):
            sentences = self._split_sentences(paragraph)
            if not sentences:
                continue

            current: list[str] = [sentences[0]]
            current_tokens = set(self._normalize_tokens(sentences[0]))

            for sentence in sentences[1:]:
                sentence_tokens = set(self._normalize_tokens(sentence))
                similarity = self._overlap_similarity(current_tokens, sentence_tokens)
                if similarity < (1.0 - self.config.semantic_break_threshold):
                    segments.append(" ".join(current).strip())
                    current = [sentence]
                    current_tokens = set(self._normalize_tokens(sentence))
                else:
                    current.append(sentence)
                    current_tokens |= sentence_tokens

            if current:
                segments.append(" ".join(current).strip())
        return [segment for segment in segments if segment]

    def _split_long_segment(
        self, document: ParsedDocument, segment: str, start_index: int
    ) -> list["_ChunkWithTokens"]:
        tokens = self._tokenize(segment)
        stride = self.config.max_tokens - self.config.overlap_tokens
        chunks: list[_ChunkWithTokens] = []
        i = 0
        chunk_index = start_index

        while i < len(tokens):
            window_tokens = tokens[i : i + self.config.max_tokens]
            window_text = self._detokenize(window_tokens)
            chunk = _ChunkWithTokens(
                chunk_id=f"{document.doc_id}-chunk-{chunk_index:04d}",
                doc_id=document.doc_id,
                text=window_text,
                text_tokens=window_tokens,
                token_count=len(window_tokens),
                metadata={
                    **document.metadata,
                    "chunk_index": chunk_index,
                    "window_start": i,
                    "window_end": i + len(window_tokens),
                },
            )
            chunks.append(chunk)
            chunk_index += 1
            if len(window_tokens) < self.config.max_tokens:
                break
            i += stride

        return chunks

    def _finalize_chunk(
        self, document: ParsedDocument, state: _ChunkState, chunk_index: int
    ) -> "_ChunkWithTokens":
        text = "\n\n".join(s for s in state.segments if s.strip()).strip()
        tokens = self._tokenize(text)
        return _ChunkWithTokens(
            chunk_id=f"{document.doc_id}-chunk-{chunk_index:04d}",
            doc_id=document.doc_id,
            text=text,
            text_tokens=tokens,
            token_count=len(tokens),
            metadata={**document.metadata, "chunk_index": chunk_index},
        )

    @staticmethod
    def _split_paragraphs(text: str) -> list[str]:
        return [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]

    @staticmethod
    def _split_sentences(paragraph: str) -> list[str]:
        return [part.strip() for part in _SENTENCE_SPLIT.split(paragraph) if part.strip()]

    @staticmethod
    def _normalize_tokens(text: str) -> list[str]:
        return [token.lower() for token in _TOKEN_PATTERN.findall(text)]

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return _TOKEN_PATTERN.findall(text)

    @staticmethod
    def _detokenize(tokens: list[str]) -> str:
        """Rebuild readable text from token stream.

        This function is intentionally simple and deterministic for tests.
        It inserts spaces between alphanumeric tokens while keeping punctuation
        adjacent to the preceding token where possible.
        """

        if not tokens:
            return ""
        output: list[str] = [tokens[0]]
        for token in tokens[1:]:
            if re.match(r"\w", token) and re.match(r"\w", output[-1][-1]):
                output.append(" " + token)
            elif re.match(r"\w", token) and output[-1] and output[-1][-1] in {".", ",", ";", ":", "!", "?"}:
                output.append(" " + token)
            elif token in {"(", "[", "{"}:
                output.append(" " + token)
            elif output[-1] in {"(", "[", "{"}:
                output.append(token)
            else:
                output.append(token)
        return "".join(output)

    @staticmethod
    def _overlap_similarity(a: set[str], b: set[str]) -> float:
        if not a or not b:
            return 0.0
        return len(a & b) / len(a | b)


@dataclass(slots=True)
class _ChunkWithTokens:
    chunk_id: str
    doc_id: str
    text: str
    text_tokens: list[str]
    token_count: int
    metadata: dict[str, object]

    def fingerprint(self) -> str:
        return sha1(self.text.encode("utf-8")).hexdigest()
