from __future__ import annotations

import os
from typing import Any

from deepagents import create_deep_agent
from langchain_core.language_models.chat_models import BaseChatModel


def _find_agents_md() -> str:
    dirs = [
        os.getcwd(),
        os.path.dirname(os.path.abspath(__file__)),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."),
    ]
    for d in dirs:
        candidate = os.path.join(d, "AGENTS.md")
        if os.path.isfile(candidate):
            return os.path.relpath(candidate)
    return "./AGENTS.md"


def create_weather_agent(
    model: BaseChatModel,
    tools: list,
    memory: list[str] | None = None,
) -> Any:
    agents_md = _find_agents_md()
    return create_deep_agent(
        model=model,
        memory=memory or [agents_md],
        tools=tools,
        subagents=[],
    )
