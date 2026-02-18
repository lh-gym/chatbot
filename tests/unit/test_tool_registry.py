import pytest
from pydantic import BaseModel, Field, ValidationError

from rag_agent.agent.registry import ToolRegistry, ToolSpec


class EchoInput(BaseModel):
    value: int = Field(ge=1)


def test_tool_registry_validation() -> None:
    registry = ToolRegistry()

    def _handler(data: EchoInput) -> str:
        return str(data.value)

    registry.register(
        ToolSpec(
            name="echo",
            description="echo positive int",
            args_schema=EchoInput,
            handler=_handler,
        )
    )

    assert registry.execute("echo", {"value": 3}) == "3"

    with pytest.raises(ValidationError):
        registry.execute("echo", {"value": 0})


def test_duplicate_tool_registration_rejected() -> None:
    registry = ToolRegistry()

    def _handler(data: EchoInput) -> str:
        return str(data.value)

    spec = ToolSpec(
        name="echo",
        description="echo positive int",
        args_schema=EchoInput,
        handler=_handler,
    )

    registry.register(spec)
    with pytest.raises(ValueError):
        registry.register(spec)
