from rag_agent.config import ChunkingConfig
from rag_agent.ingest.chunker import SemanticSlidingChunker
from rag_agent.types import ParsedDocument


def _make_long_text(token_count: int = 2600) -> str:
    sentence = "Data governance requires strict access control and encryption."
    words = sentence.split()
    repeated = []
    while len(repeated) < token_count:
        repeated.extend(words)
    body = " ".join(repeated[:token_count])
    return body + "\n\n" + body


def test_chunker_token_bounds_and_overlap() -> None:
    config = ChunkingConfig(min_tokens=500, max_tokens=1200, overlap_tokens=120)
    chunker = SemanticSlidingChunker(config)
    doc = ParsedDocument(doc_id="doc-1", text=_make_long_text(), metadata={"source": "unit"})

    chunks = chunker.chunk_document(doc)

    assert len(chunks) >= 2
    assert all(chunk.token_count <= 1200 for chunk in chunks)
    assert all(chunk.token_count >= 500 for chunk in chunks[:-1])

    first_tokens = chunker._tokenize(chunks[0].text)
    second_tokens = chunker._tokenize(chunks[1].text)
    expected_overlap = first_tokens[-120:]

    assert second_tokens[:120] == expected_overlap
