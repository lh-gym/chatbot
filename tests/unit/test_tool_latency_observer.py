from pydantic import BaseModel

from rag_agent.agent.registry import ToolRegistry, ToolSpec


class EchoInput(BaseModel):
    text: str


def test_tool_observer_captures_latency_and_payload() -> None:
    registry = ToolRegistry()

    def _handler(data: EchoInput) -> str:
        return data.text.upper()

    registry.register(
        ToolSpec(
            name="echo",
            description="uppercase",
            args_schema=EchoInput,
            handler=_handler,
        )
    )

    observed = []
    registry.set_observer(observed.append)
    result = registry.execute("echo", {"text": "hello"})
    registry.set_observer(None)

    assert result == "HELLO"
    assert len(observed) == 1
    assert observed[0].name == "echo"
    assert observed[0].input_payload == {"text": "hello"}
    assert observed[0].latency_ms >= 0.0
