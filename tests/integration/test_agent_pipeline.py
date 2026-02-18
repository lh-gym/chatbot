import re
from dataclasses import dataclass

from rag_agent.agent.planner import RagAgentPlanner
from rag_agent.agent.registry import ToolRegistry
from rag_agent.agent.tools import register_builtin_tools
from rag_agent.ingest.embedder import HashingEmbedder
from rag_agent.obs.tracing import TraceStore
from rag_agent.retrieval.fusion import FusionLayer
from rag_agent.retrieval.retriever import DualRouteRetriever
from rag_agent.retrieval.vector_store import InMemoryVectorStore
from rag_agent.types import DocumentChunk


class MockLLM:
    pass


@dataclass(slots=True)
class _DummyAction:
    tool: str
    tool_input: dict[str, object]


class _MockExecutor:
    def __init__(self, registry: ToolRegistry) -> None:
        self.registry = registry

    def invoke(self, payload: dict[str, object]) -> dict[str, object]:
        question = str(payload["input"])
        search_output = self.registry.execute(
            "internal_search", {"query": question, "top_k": 1}
        )
        match = re.search(r"\[([^\]]+)\]", search_output)
        citation = match.group(1) if match else "missing"
        answer = f"Policy requires encryption of customer data at rest [{citation}]."
        return {
            "output": answer,
            "intermediate_steps": [
                (_DummyAction(tool="internal_search", tool_input={"query": question}), search_output)
            ],
        }


def test_agent_selects_internal_search_and_returns_citations(tmp_path) -> None:
    embedder = HashingEmbedder()
    vector_store = InMemoryVectorStore()
    chunks = [
        DocumentChunk(
            chunk_id="policy-doc-chunk-0000",
            doc_id="policy-doc",
            text="Company policy states all employees must encrypt customer data at rest.",
            token_count=12,
            metadata={"source": "policy"},
        ),
        DocumentChunk(
            chunk_id="faq-doc-chunk-0000",
            doc_id="faq-doc",
            text="Holiday arrangements are documented in the employee handbook.",
            token_count=10,
            metadata={"source": "faq"},
        ),
    ]
    vector_store.upsert(chunks, embedder.embed_documents([c.text for c in chunks]))

    retriever = DualRouteRetriever(vector_store, embedder, FusionLayer())
    registry = ToolRegistry()
    register_builtin_tools(registry, retriever, sqlite_path=str(tmp_path / "kv.db"))

    trace_store = TraceStore()
    planner = RagAgentPlanner(
        llm=MockLLM(),
        tool_registry=registry,
        trace_store=trace_store,
        executor=_MockExecutor(registry),
    )

    result = planner.invoke("What does policy require for customer data?")

    assert result["citations"] == ["policy-doc-chunk-0000"]
    assert result["groundedness"] >= 0.95

    trace = trace_store.get(result["trace_id"])
    assert any(tool.name == "internal_search" for tool in trace.tool_traces)
    assert any("encrypt customer data" in source for source in trace.source_snippets)
