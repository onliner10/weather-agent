from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage

from weather_agent.application.conversation_service import build_conversation_messages


def test_build_conversation_messages_uses_persisted_turns_before_current_message() -> None:
    messages = build_conversation_messages(
        [
            {"role": "user", "text": "Jaka była prognoza?"},
            {"role": "bot", "text": "W Warszawie było 18 stopni."},
            {"role": "bot", "answer_summary": "Będzie padać po 18."},
            {"role": "system", "text": "ignored"},
            {"role": "user", "text": ""},
        ],
        "A jutro?",
    )

    assert [type(message) for message in messages] == [
        HumanMessage,
        AIMessage,
        AIMessage,
        HumanMessage,
    ]
    assert [message.content for message in messages] == [
        "Jaka była prognoza?",
        "W Warszawie było 18 stopni.",
        "Będzie padać po 18.",
        "A jutro?",
    ]
