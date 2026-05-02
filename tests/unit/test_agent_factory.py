from __future__ import annotations

from typing import Any

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import StructuredTool

from weather_agent.agent_factory import (
    AgentRuntime,
    _load_weather_agent_prompt,
    create_weather_agent,
)


def test_weather_agent_prompt_is_loaded_from_runtime_prompt_file() -> None:
    result = _load_weather_agent_prompt()

    assert "Pogodowy Asystent" in result
    assert "Jesteś polskim asystentem pogodowym" in result
    assert "Weather Agent Repository Instructions" not in result


class _FakeToolCallingModel:
    def __init__(self, responses: list[AIMessage]) -> None:
        self._responses = responses
        self.calls: list[list[BaseMessage]] = []
        self.configs: list[dict[str, Any] | None] = []
        self.bound_tools: list[Any] = []

    def bind_tools(self, tools: list[Any]) -> _FakeToolCallingModel:
        self.bound_tools = tools
        return self

    def invoke(
        self,
        messages: list[BaseMessage],
        config: dict[str, Any] | None = None,
    ) -> AIMessage:
        self.calls.append(list(messages))
        self.configs.append(config)
        return self._responses.pop(0)


def test_create_weather_agent_returns_plain_runtime_without_deepagents() -> None:
    runtime = create_weather_agent(
        model=_FakeToolCallingModel([AIMessage(content="ok")]),  # type: ignore[arg-type]
        tools=[],
        system_prompt_suffix="suffix",
    )

    assert isinstance(runtime, AgentRuntime)


def test_agent_runtime_calls_tool_and_returns_final_answer() -> None:
    def get_weather(location: str) -> str:
        """Return test weather."""
        return f"Pogoda dla {location}: 12 C"

    tool = StructuredTool.from_function(get_weather)
    model = _FakeToolCallingModel(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "get_weather",
                        "args": {"location": "Warszawa"},
                        "id": "call-1",
                    }
                ],
            ),
            AIMessage(content="W Warszawie jest 12 C."),
        ]
    )
    runtime = AgentRuntime(
        model=model,  # type: ignore[arg-type]
        tools=[tool],
        system_prompt="System",
    )

    result = runtime.invoke(
        {"messages": [HumanMessage(content="Jaka pogoda?")]},
        config={"metadata": {"trace": "yes"}},
    )

    assert result["messages"][-1].content == "W Warszawie jest 12 C."
    assert model.configs == [{"metadata": {"trace": "yes"}}, {"metadata": {"trace": "yes"}}]
    assert isinstance(model.calls[0][0], SystemMessage)
    assert isinstance(model.calls[1][-1], ToolMessage)
    assert "Pogoda dla Warszawa" in str(model.calls[1][-1].content)
