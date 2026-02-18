"""Tool registry built on Pydantic v2 models."""

from __future__ import annotations

from collections.abc import Callable
from time import perf_counter
from typing import Any

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, ConfigDict, Field

from rag_agent.types import ToolTrace


class ToolSpec(BaseModel):
    """Declarative tool specification for registration and validation."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    description: str
    args_schema: type[BaseModel]
    handler: Callable[[BaseModel], str]
    tags: list[str] = Field(default_factory=list)

    def invoke(self, payload: dict[str, Any]) -> str:
        data = self.args_schema.model_validate(payload)
        return self.handler(data)


class ToolRegistry:
    """Stores tool specs and exports LangChain-compatible tool objects."""

    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}
        self._observer: Callable[[ToolTrace], None] | None = None

    def register(self, spec: ToolSpec) -> None:
        if spec.name in self._tools:
            raise ValueError(f"Tool already registered: {spec.name}")
        self._tools[spec.name] = spec

    def set_observer(self, observer: Callable[[ToolTrace], None] | None) -> None:
        """Set an optional callback invoked after each tool execution."""
        self._observer = observer

    def execute(self, name: str, payload: dict[str, Any]) -> str:
        spec = self._tools.get(name)
        if spec is None:
            raise KeyError(f"Unknown tool: {name}")
        return self._execute_spec(spec, payload)

    def as_langchain_tools(self) -> list[StructuredTool]:
        tools: list[StructuredTool] = []
        for spec in self._tools.values():
            tools.append(
                StructuredTool.from_function(
                    name=spec.name,
                    description=spec.description,
                    args_schema=spec.args_schema,
                    func=self._build_function(spec),
                )
            )
        return tools

    def specs(self) -> list[ToolSpec]:
        return list(self._tools.values())

    def _build_function(self, spec: ToolSpec) -> Callable[..., str]:
        def _callable(**kwargs: Any) -> str:
            return self._execute_spec(spec, kwargs)

        return _callable

    def _execute_spec(self, spec: ToolSpec, payload: dict[str, Any]) -> str:
        start = perf_counter()
        output = spec.invoke(payload)
        latency_ms = (perf_counter() - start) * 1000.0

        if self._observer is not None:
            self._observer(
                ToolTrace(
                    name=spec.name,
                    input_payload=payload,
                    output_preview=output[:320],
                    latency_ms=latency_ms,
                )
            )
        return output
