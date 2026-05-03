from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

from langchain_core.messages import AIMessage, HumanMessage

from weather_agent.adapters.telegram import handler
from weather_agent.application.conversation_models import BotAttachment, BotReply, UserMessage
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


async def test_message_handler_sends_text_and_png_attachment(monkeypatch) -> None:
    captured_request: UserMessage | None = None

    class FakeConversationService:
        def __init__(self, **_kwargs: object) -> None:
            pass

        async def handle_reply(self, request: UserMessage) -> BotReply:
            nonlocal captured_request
            captured_request = request
            return BotReply(
                text="Dołączam wykres.",
                attachments=(
                    BotAttachment(
                        filename="prognoza.png",
                        media_type="image/png",
                        data=b"\x89PNG\r\n\x1a\nfake",
                    ),
                ),
            )

    monkeypatch.setattr(handler, "ConversationService", FakeConversationService)
    container = MagicMock()
    container.settings.model.timeout_seconds = 3
    message_handler = await handler.make_message_handler(container)

    update = MagicMock()
    update.effective_user.id = 42
    update.effective_chat.id = 100
    update.message.text = "Pokaż wykres wiatru"
    update.message.message_id = 10
    update.message.message_thread_id = None
    update.message.reply_to_message = None
    update.message.reply_text = AsyncMock()
    update.message.reply_photo = AsyncMock()
    context = MagicMock()

    await message_handler(update, context)

    assert captured_request is not None
    assert captured_request.text == "Pokaż wykres wiatru"
    update.message.reply_text.assert_awaited_once_with("Dołączam wykres.")
    update.message.reply_photo.assert_awaited_once()
    assert update.message.reply_photo.call_args.kwargs["caption"] is None
