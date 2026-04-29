"""LangSmith tracing utilities for conversational runtime boundaries."""

from __future__ import annotations

from typing import Any

from weather_agent.graphs.state import ConversationState


def build_telegram_turn_metadata(state: ConversationState) -> dict[str, Any]:
    """Extract concise trace metadata from a conversation state.

    Omits secrets and large raw payloads.  Includes identifiers,
    intent, and resolved location/time summaries.
    """
    metadata: dict[str, Any] = {
        "context_key": state.get("context_key"),
        "chat_id": state.get("chat_id"),
        "message_thread_id": state.get("message_thread_id"),
        "telegram_user_id": state.get("authorized_user_id"),
        "inbound_message_id": state.get("message_id"),
        "reply_to_message_id": state.get("reply_to_message_id"),
        "is_reply_follow_up": state.get("reply_to_message_id") is not None,
        "resolved_intent": state.get("resolved_intent"),
        "user_message_preview": ((state.get("user_message") or "")[:80] or None),
    }

    loc = state.get("resolved_location")
    if loc is not None:
        metadata["resolved_location_name"] = loc.name
        metadata["resolved_location_id"] = loc.id

    tr = state.get("resolved_time_range")
    if tr is not None:
        metadata["resolved_time_explanation"] = tr.explanation

    return {k: v for k, v in metadata.items() if v is not None}


def build_telegram_turn_tags(state: ConversationState) -> list[str]:
    """Build LangSmith tags for a Telegram turn."""
    tags = ["telegram", "conversation"]
    intent = state.get("resolved_intent")
    if intent:
        tags.append(f"intent:{intent}")
    if state.get("reply_to_message_id") is not None:
        tags.append("reply-follow-up")
    return tags


def build_run_name(state: ConversationState) -> str:
    """Build a stable run name for a Telegram turn."""
    context_key = state.get("context_key", "unknown")
    intent = state.get("resolved_intent") or "unknown"
    return f"telegram-turn:{context_key}:{intent}"


def build_node_metadata(
    state: ConversationState,
    node_name: str,
) -> dict[str, Any]:
    """Build metadata for an individual graph node span.

    Inherits the conversation context so every node is searchable
    in LangSmith without expanding the parent trace.
    """
    metadata = build_telegram_turn_metadata(state)
    metadata["node"] = node_name
    return metadata


def build_graph_config(state: ConversationState) -> dict[str, Any]:
    """Build a RunnableConfig dict for LangGraph invocation.

    This config is passed as the *config* argument to
    ``CompiledStateGraph.ainvoke()`` so LangSmith auto-tracing
    picks up the run name, tags, and metadata.
    """
    return {
        "run_name": build_run_name(state),
        "tags": build_telegram_turn_tags(state),
        "metadata": build_telegram_turn_metadata(state),
    }
