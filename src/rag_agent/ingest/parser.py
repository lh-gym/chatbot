"""Parsing interfaces and concrete parsers for heterogeneous inputs."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from rag_agent.types import ParsedDocument


class Parser(ABC):
    """Base parser interface used by the ingest pipeline."""

    extensions: tuple[str, ...] = ()

    @abstractmethod
    def parse(self, path: Path, *, doc_id: str | None = None) -> ParsedDocument:
        """Parse a file into normalized text + metadata."""


class TextParser(Parser):
    """Parser for plain text documents."""

    extensions = (".txt", ".log")

    def parse(self, path: Path, *, doc_id: str | None = None) -> ParsedDocument:
        text = path.read_text(encoding="utf-8")
        return ParsedDocument(
            doc_id=doc_id or path.stem,
            text=text,
            metadata={"source": str(path), "format": "text"},
        )


class MarkdownParser(Parser):
    """Parser for markdown documents."""

    extensions = (".md", ".markdown")

    def parse(self, path: Path, *, doc_id: str | None = None) -> ParsedDocument:
        text = path.read_text(encoding="utf-8")
        return ParsedDocument(
            doc_id=doc_id or path.stem,
            text=text,
            metadata={"source": str(path), "format": "markdown"},
        )


class JsonParser(Parser):
    """Parser for JSON documents with deterministic normalization."""

    extensions = (".json",)

    def parse(self, path: Path, *, doc_id: str | None = None) -> ParsedDocument:
        payload: Any = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            text = json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2)
            metadata: dict[str, Any] = {
                "source": str(path),
                "format": "json",
                "keys": sorted(payload.keys()),
            }
        elif isinstance(payload, list):
            text = json.dumps(payload, ensure_ascii=False, indent=2)
            metadata = {
                "source": str(path),
                "format": "json",
                "length": len(payload),
            }
        else:
            text = str(payload)
            metadata = {"source": str(path), "format": "json"}
        return ParsedDocument(doc_id=doc_id or path.stem, text=text, metadata=metadata)


class ParserRegistry:
    """Maps file extension to parser implementation."""

    def __init__(self, parsers: list[Parser] | None = None) -> None:
        self._parsers: dict[str, Parser] = {}
        for parser in parsers or [TextParser(), MarkdownParser(), JsonParser()]:
            self.register(parser)

    def register(self, parser: Parser) -> None:
        for extension in parser.extensions:
            self._parsers[extension.lower()] = parser

    def parse_path(self, path: str | Path, *, doc_id: str | None = None) -> ParsedDocument:
        file_path = Path(path)
        parser = self._parsers.get(file_path.suffix.lower())
        if parser is None:
            raise ValueError(f"No parser registered for extension: {file_path.suffix}")
        return parser.parse(file_path, doc_id=doc_id)
