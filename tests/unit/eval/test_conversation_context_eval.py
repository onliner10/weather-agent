from __future__ import annotations

from weather_agent.application.conversation_service import build_conversation_messages


def test_follow_up_eval_context_includes_previous_user_and_assistant_turn() -> None:
    messages = build_conversation_messages(
        [
            {"role": "user", "text": "Jaka będzie jutro pogoda w Warszawie?"},
            {"role": "bot", "text": "Jutro w Warszawie będzie 18 C i słaby wiatr."},
        ],
        "A wiatr?",
    )

    assert [message.type for message in messages] == ["human", "ai", "human"]
    assert [message.content for message in messages] == [
        "Jaka będzie jutro pogoda w Warszawie?",
        "Jutro w Warszawie będzie 18 C i słaby wiatr.",
        "A wiatr?",
    ]


def test_follow_up_eval_context_uses_answer_summary_fallback() -> None:
    messages = build_conversation_messages(
        [
            {"role": "user", "text": "Czy będzie padać wieczorem?"},
            {"role": "bot", "answer_summary": "Po 18 spodziewany jest deszcz."},
        ],
        "A rano?",
    )

    assert [message.content for message in messages] == [
        "Czy będzie padać wieczorem?",
        "Po 18 spodziewany jest deszcz.",
        "A rano?",
    ]
