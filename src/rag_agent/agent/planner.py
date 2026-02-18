"""LangChain-based agent planner with observability hooks."""

from __future__ import annotations

import re
from importlib import import_module
from typing import Any

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

_agents_module = import_module("langchain.agents")
create_agent = getattr(_agents_module, "create_agent", None)
AgentExecutor = getattr(_agents_module, "AgentExecutor", None)
create_tool_calling_agent = getattr(_agents_module, "create_tool_calling_agent", None)
_AGENT_RUNTIME = (
    "legacy"
    if callable(AgentExecutor) and callable(create_tool_calling_agent)
    else "graph"
)

from rag_agent.agent.registry import ToolRegistry
from rag_agent.config import AgentConfig
from rag_agent.obs.tracing import Timer, TraceStore, estimate_token_count
from rag_agent.types import ToolTrace

_SYSTEM_PROMPT = """
You are a production RAG assistant.

Rules:
1) Always ground answers in tool outputs from `internal_search`.
2) Cite every factual statement using chunk citations like [doc-1-chunk-0003].
3) If evidence is missing, explicitly say you cannot verify.
4) Prefer concise, accurate answers with attributed evidence.

Quality targets:
- Groundedness must remain above 95%.
- Keep response latency low and avoid unnecessary tool calls.
""".strip()


class RagAgentPlanner:
    """High-level orchestrator wrapping a LangChain tool-calling agent."""

    def __init__(
        self,
        *,
        llm: Any,
        tool_registry: ToolRegistry,
        trace_store: TraceStore,
        config: AgentConfig | None = None,
        executor: Any | None = None,
    ) -> None:
        self.llm = llm
        self.tool_registry = tool_registry
        self.trace_store = trace_store
        self.config = config or AgentConfig()

        self.tools = self.tool_registry.as_langchain_tools()
        self._runtime = "custom" if executor is not None else _AGENT_RUNTIME
        if executor is not None:
            self.executor = executor
        elif _AGENT_RUNTIME == "legacy":
            if not callable(create_tool_calling_agent) or not callable(AgentExecutor):
                raise RuntimeError("Legacy LangChain agent runtime is unavailable.")
            prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", _SYSTEM_PROMPT),
                    MessagesPlaceholder(variable_name="chat_history", optional=True),
                    ("human", "{input}"),
                    MessagesPlaceholder(variable_name="agent_scratchpad"),
                ]
            )

            agent = create_tool_calling_agent(self.llm, self.tools, prompt)
            self.executor = AgentExecutor(
                agent=agent,
                tools=self.tools,
                max_iterations=self.config.max_iterations,
                return_intermediate_steps=True,
                verbose=False,
                handle_parsing_errors=True,
            )
        else:
            if not callable(create_agent):
                raise RuntimeError("LangChain create_agent is unavailable.")
            self.executor = create_agent(
                model=self.llm,
                tools=self.tools,
                system_prompt=_SYSTEM_PROMPT,
            )

    def invoke(
        self,
        question: str,
        *,
        chat_history: list[Any] | None = None,
    ) -> dict[str, Any]:
        """Run one full agent turn and persist trace metrics.

        Returns:
            A structured payload containing answer, citations, trace id, latency,
            groundedness score, and whether the latency target (<8s by default)
            has been satisfied.
        """

        observed_tools: list[ToolTrace] = []
        self.tool_registry.set_observer(observed_tools.append)
        try:
            with Timer() as timer:
                if self._runtime == "graph":
                    messages = list(chat_history or [])
                    messages.append({"role": "user", "content": question})
                    result = self.executor.invoke({"messages": messages})
                else:
                    result = self.executor.invoke(
                        {
                            "input": question,
                            "chat_history": chat_history or [],
                        }
                    )
        finally:
            self.tool_registry.set_observer(None)

        if self._runtime == "graph":
            answer = _extract_graph_answer(result)
            intermediate_steps = _extract_graph_tool_steps(result)
        else:
            answer = str(result.get("output", ""))
            intermediate_steps = result.get("intermediate_steps", [])
        fallback_traces, source_snippets = self._extract_tool_traces(intermediate_steps)
        tool_traces = observed_tools or fallback_traces
        citations = _extract_citations(answer)

        record = self.trace_store.create_record(
            question=question,
            answer=answer,
            citations=citations,
            source_snippets=source_snippets,
            tool_traces=tool_traces,
            input_tokens=estimate_token_count(question),
            output_tokens=estimate_token_count(answer),
            latency_ms=timer.elapsed_ms,
        )

        return {
            "answer": answer,
            "citations": citations,
            "trace_id": record.trace_id,
            "latency_ms": record.latency_ms,
            "latency_target_met": record.latency_ms
            <= (self.config.target_latency_seconds * 1000.0),
            "groundedness": record.groundedness,
            "groundedness_target_met": record.groundedness
            >= self.config.groundedness_target,
        }

    def _extract_tool_traces(
        self, intermediate_steps: list[Any]
    ) -> tuple[list[ToolTrace], list[str]]:
        traces: list[ToolTrace] = []
        source_snippets: list[str] = []

        for step in intermediate_steps:
            if not isinstance(step, tuple) or len(step) != 2:
                continue
            action, observation = step
            tool_name = str(getattr(action, "tool", "unknown"))
            tool_input = getattr(action, "tool_input", {})
            observation_text = str(observation)
            traces.append(
                ToolTrace(
                    name=tool_name,
                    input_payload=tool_input if isinstance(tool_input, dict) else {"raw": str(tool_input)},
                    output_preview=observation_text[:320],
                    latency_ms=0.0,
                )
            )
            if tool_name == "internal_search":
                source_snippets.extend(_extract_search_snippets(observation_text))

        return traces, source_snippets


def _extract_citations(answer: str) -> list[str]:
    citations = re.findall(r"\[([^\]]+)\]", answer)
    deduped: list[str] = []
    for citation in citations:
        if citation not in deduped:
            deduped.append(citation)
    return deduped


def _extract_graph_answer(result: Any) -> str:
    if not isinstance(result, dict):
        return str(result)
    messages = result.get("messages", [])
    if not isinstance(messages, list) or not messages:
        return str(result.get("output", ""))
    last = messages[-1]
    if isinstance(last, dict):
        return str(last.get("content", ""))
    content = getattr(last, "content", "")
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and "text" in item:
                parts.append(str(item["text"]))
            else:
                parts.append(str(item))
        return " ".join(parts).strip()
    return str(content)


def _extract_graph_tool_steps(result: Any) -> list[Any]:
    if not isinstance(result, dict):
        return []
    messages = result.get("messages", [])
    if not isinstance(messages, list):
        return []

    steps: list[Any] = []
    for message in messages:
        msg_type = str(getattr(message, "type", "")).lower()
        if "tool" not in msg_type and message.__class__.__name__.lower() != "toolmessage":
            continue

        tool_name = str(getattr(message, "name", "unknown"))
        tool_content = str(getattr(message, "content", ""))
        action = type(
            "ToolAction",
            (),
            {"tool": tool_name, "tool_input": {"raw": "graph-runtime"}},
        )()
        steps.append((action, tool_content))
    return steps


def _extract_search_snippets(tool_output: str) -> list[str]:
    snippets: list[str] = []
    for line in tool_output.splitlines():
        line = line.strip()
        if not line:
            continue
        # Expected format: [chunk-id] score=... snippet
        right = line.split(" ", 2)
        if len(right) == 3:
            snippets.append(right[2])
        else:
            snippets.append(line)
    return snippets
