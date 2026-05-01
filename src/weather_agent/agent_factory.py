from __future__ import annotations

from datetime import datetime
from importlib import resources
from typing import Any
from zoneinfo import ZoneInfo

from deepagents import create_deep_agent
from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.checkpoint.memory import MemorySaver

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


def create_weather_agent(
    model: BaseChatModel,
    tools: list[Any],
    system_prompt_suffix: str = "",
    checkpointer: Any | None = None,
) -> Any:
    base_prompt = _load_weather_agent_prompt()

    system_prompt = base_prompt
    if system_prompt_suffix:
        system_prompt = f"{base_prompt}\n\n{system_prompt_suffix}"

    cp = checkpointer if checkpointer is not None else MemorySaver()

    return create_deep_agent(
        model=model,
        tools=tools,
        system_prompt=system_prompt,
        checkpointer=cp,
        subagents=[],
    )
