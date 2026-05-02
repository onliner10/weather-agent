from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from datetime import datetime
from importlib import resources
from typing import Any
from zoneinfo import ZoneInfo

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool

from weather_agent.observability.logging import get_logger

logger = get_logger(__name__)

_WEATHER_AGENT_PROMPT: str | None = None
_WARSAW_TZ = ZoneInfo("Europe/Warsaw")


def _load_weather_agent_prompt() -> str:
    global _WEATHER_AGENT_PROMPT
    if _WEATHER_AGENT_PROMPT is not None:
        return _WEATHER_AGENT_PROMPT

    prompt_path = resources.files("weather_agent.llm.prompts").joinpath("weather_agent.md")
    _WEATHER_AGENT_PROMPT = prompt_path.read_text(encoding="utf-8")
    logger.info("weather_agent_prompt_loaded", path=str(prompt_path))
    return _WEATHER_AGENT_PROMPT


def build_current_time_prompt_suffix(now: datetime | None = None) -> str:
    effective_now = datetime.now(_WARSAW_TZ) if now is None else now.astimezone(_WARSAW_TZ)
    return (
        "Bieżąca data i godzina w strefie Europe/Warsaw: "
        f"{effective_now.strftime('%Y-%m-%d %H:%M')}."
    )


def _tool_result_to_content(result: object) -> str:
    if isinstance(result, str):
        return result
    if hasattr(result, "model_dump"):
        result = result.model_dump()
    return json.dumps(result, ensure_ascii=False, default=str)


class AgentRuntime:
    def __init__(
        self,
        model: BaseChatModel,
        tools: Sequence[BaseTool],
        system_prompt: str,
        max_tool_rounds: int = 8,
    ) -> None:
        self._model = model.bind_tools(tools) if tools else model
        self._tools = {tool.name: tool for tool in tools}
        self._system_prompt = system_prompt
        self._max_tool_rounds = max_tool_rounds

    def invoke(
        self,
        payload: dict[str, Any],
        config: RunnableConfig | None = None,
    ) -> dict[str, list[BaseMessage]]:
        result_messages, model_messages = self._initial_messages(payload)
        for _ in range(self._max_tool_rounds):
            response = self._model.invoke(model_messages, config=config)
            if not isinstance(response, AIMessage):
                response = AIMessage(content=str(response))
            result_messages.append(response)
            model_messages.append(response)
            tool_calls = response.tool_calls
            if not tool_calls:
                return {"messages": result_messages}
            for tool_call in tool_calls:
                tool_message = self._invoke_tool(tool_call)
                result_messages.append(tool_message)
                model_messages.append(tool_message)
        return {
            "messages": [
                *result_messages,
                AIMessage(content="Przepraszam, nie udało się zakończyć obsługi narzędzi."),
            ]
        }

    async def ainvoke(
        self,
        payload: dict[str, Any],
        config: RunnableConfig | None = None,
    ) -> dict[str, list[BaseMessage]]:
        result_messages, model_messages = self._initial_messages(payload)
        for _ in range(self._max_tool_rounds):
            response = await self._model.ainvoke(model_messages, config=config)
            if not isinstance(response, AIMessage):
                response = AIMessage(content=str(response))
            result_messages.append(response)
            model_messages.append(response)
            tool_calls = response.tool_calls
            if not tool_calls:
                return {"messages": result_messages}
            for tool_call in tool_calls:
                tool_message = await self._ainvoke_tool(tool_call)
                result_messages.append(tool_message)
                model_messages.append(tool_message)
        return {
            "messages": [
                *result_messages,
                AIMessage(content="Przepraszam, nie udało się zakończyć obsługi narzędzi."),
            ]
        }

    def _initial_messages(
        self,
        payload: dict[str, Any],
    ) -> tuple[list[BaseMessage], list[BaseMessage]]:
        raw_messages = payload.get("messages", [])
        messages = list(raw_messages) if isinstance(raw_messages, list) else []
        return messages, [SystemMessage(content=self._system_prompt), *messages]

    def _invoke_tool(self, tool_call: Mapping[str, Any]) -> ToolMessage:
        tool_name = str(tool_call.get("name", ""))
        tool_call_id = str(tool_call.get("id", tool_name))
        tool = self._tools.get(tool_name)
        if tool is None:
            return ToolMessage(
                content=f"Unknown tool: {tool_name}",
                tool_call_id=tool_call_id,
                status="error",
            )
        try:
            result = tool.invoke(tool_call.get("args", {}))
            return ToolMessage(
                content=_tool_result_to_content(result),
                tool_call_id=tool_call_id,
            )
        except Exception as exc:
            return ToolMessage(
                content=f"Tool error: {type(exc).__name__}: {exc}",
                tool_call_id=tool_call_id,
                status="error",
            )

    async def _ainvoke_tool(self, tool_call: Mapping[str, Any]) -> ToolMessage:
        tool_name = str(tool_call.get("name", ""))
        tool_call_id = str(tool_call.get("id", tool_name))
        tool = self._tools.get(tool_name)
        if tool is None:
            return ToolMessage(
                content=f"Unknown tool: {tool_name}",
                tool_call_id=tool_call_id,
                status="error",
            )
        try:
            result = await tool.ainvoke(tool_call.get("args", {}))
            return ToolMessage(
                content=_tool_result_to_content(result),
                tool_call_id=tool_call_id,
            )
        except Exception as exc:
            return ToolMessage(
                content=f"Tool error: {type(exc).__name__}: {exc}",
                tool_call_id=tool_call_id,
                status="error",
            )


def create_weather_agent(
    model: BaseChatModel,
    tools: Sequence[BaseTool],
    system_prompt_suffix: str = "",
) -> AgentRuntime:
    base_prompt = _load_weather_agent_prompt()

    system_prompt = base_prompt
    if system_prompt_suffix:
        system_prompt = f"{base_prompt}\n\n{system_prompt_suffix}"

    return AgentRuntime(
        model=model,
        tools=tools,
        system_prompt=system_prompt,
    )
