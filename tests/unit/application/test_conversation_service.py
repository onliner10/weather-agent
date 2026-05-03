from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from langchain_core.messages import BaseMessage
from langchain_core.runnables import RunnableConfig

from weather_agent.application.conversation_models import BotAttachment, UserMessage
from weather_agent.application.conversation_service import ConversationService, Logger
from weather_agent.llm.model_factory import ModelFactory


class _Session:
    def __init__(self) -> None:
        self.committed = False

    async def __aenter__(self) -> _Session:
        return self

    async def __aexit__(self, *_args: object) -> None:
        pass

    async def commit(self) -> None:
        self.committed = True


class _SessionFactory:
    def __init__(self) -> None:
        self.session = _Session()

    def __call__(self) -> _Session:
        return self.session


class _Memory:
    def __init__(
        self,
        *,
        pending: dict[str, Any] | None = None,
        turns: list[dict[str, Any]] | None = None,
    ) -> None:
        self.pending = pending
        self.turns = turns or []
        self.saved: list[tuple[str, dict[str, Any]]] = []

    async def get_pending_confirmation(self, context_key: str) -> dict[str, Any] | None:
        del context_key
        return self.pending

    async def save_turn(self, context_key: str, turn: dict[str, Any]) -> None:
        self.saved.append((context_key, turn))

    async def load_turns(self, context_key: str) -> list[dict[str, Any]]:
        del context_key
        return self.turns


class _KeyedMemory:
    def __init__(self) -> None:
        self.pending: dict[str, dict[str, Any]] = {}
        self.turns: dict[str, list[dict[str, Any]]] = {}

    async def get_pending_confirmation(self, context_key: str) -> dict[str, Any] | None:
        return self.pending.get(context_key)

    async def save_turn(self, context_key: str, turn: dict[str, Any]) -> None:
        self.turns.setdefault(context_key, []).append(turn)

    async def load_turns(self, context_key: str) -> list[dict[str, Any]]:
        return list(self.turns.get(context_key, []))


class _Toolbox:
    def __init__(self, tools: list[Any] | None = None) -> None:
        self._tools = tools or []

    def to_langchain_tools(self) -> list[Any]:
        return self._tools


class _ToolAnswer:
    def __init__(self, answer: str) -> None:
        self.answer = answer


class _RulesToolbox(_Toolbox):
    def __init__(self, answer: str = "Potwierdzono.", result: Any | None = None) -> None:
        super().__init__(["rules-tool"])
        self.answer = answer
        self.result = result
        self.confirmed = False
        self.cancelled = False

    async def confirm_pending_action(self) -> Any:
        self.confirmed = True
        if self.result is not None:
            return self.result
        return _ToolAnswer(self.answer)

    async def cancel_pending_action(self) -> _ToolAnswer:
        self.cancelled = True
        return _ToolAnswer("Anulowano.")


class _ModelFactory:
    provider = "fake"
    model_name = "fake-model"


def _request(text: str = "Jaka pogoda?") -> UserMessage:
    return UserMessage(
        telegram_user_id=42,
        chat_id=100,
        message_thread_id=7,
        text=text,
        message_id=10,
        reply_to_message_id=None,
    )


async def _auth_resolver(_session: object, telegram_user_id: int) -> int:
    return telegram_user_id + 1000


def _service(
    *,
    session_factory: _SessionFactory,
    memory: _Memory | _KeyedMemory,
    agent_invoker: Any,
    rules_toolbox: _RulesToolbox | None = None,
) -> ConversationService:
    return ConversationService(
        session_factory=session_factory,  # type: ignore[arg-type]
        forecast_provider=object(),
        observation_provider=object(),
        geocoder=object(),
        model_factory=_ModelFactory(),  # type: ignore[arg-type]
        rule_expression_evaluator=object(),
        timeout_seconds=3,
        memory_factory=lambda _session: memory,
        auth_resolver=_auth_resolver,  # type: ignore[arg-type]
        weather_toolbox_factory=lambda **_kwargs: _Toolbox(["weather-tool"]),
        rules_toolbox_factory=lambda **_kwargs: rules_toolbox or _RulesToolbox(),
        agent_invoker=agent_invoker,
    )


def test_user_message_context_key_includes_thread_id() -> None:
    assert _request().context_key == "100:7"
    assert (
        UserMessage(
            telegram_user_id=42,
            chat_id=100,
            message_thread_id=None,
            text="x",
            message_id=1,
            reply_to_message_id=None,
        ).context_key
        == "100"
    )


async def test_direct_confirmation_is_handled_without_agent_invocation() -> None:
    session_factory = _SessionFactory()
    memory = _Memory(pending={"action": "create_rule"})
    rules_toolbox = _RulesToolbox(answer="Reguła została utworzona.")
    agent_called = False

    async def agent_invoker(
        _model_factory: ModelFactory,
        _tools: list[Any],
        _messages: Sequence[BaseMessage],
        _config: RunnableConfig,
        _system_prompt_suffix: str,
        _timeout_seconds: float,
        _logger: Logger,
    ) -> tuple[str, bool]:
        nonlocal agent_called
        agent_called = True
        return "agent", False

    service = ConversationService(
        session_factory=session_factory,  # type: ignore[arg-type]
        forecast_provider=object(),
        observation_provider=object(),
        geocoder=object(),
        model_factory=_ModelFactory(),  # type: ignore[arg-type]
        rule_expression_evaluator=object(),
        timeout_seconds=1,
        memory_factory=lambda _session: memory,
        auth_resolver=_auth_resolver,  # type: ignore[arg-type]
        weather_toolbox_factory=lambda **_kwargs: _Toolbox(["weather-tool"]),
        rules_toolbox_factory=lambda **_kwargs: rules_toolbox,
        agent_invoker=agent_invoker,
    )

    answer = await service.handle(_request("tak"))

    assert answer == "Reguła została utworzona."
    assert rules_toolbox.confirmed is True
    assert agent_called is False
    assert session_factory.session.committed is True
    assert [turn["role"] for _key, turn in memory.saved] == ["user", "bot"]


async def test_direct_confirmation_accepts_dict_tool_result() -> None:
    session_factory = _SessionFactory()
    memory = _Memory(pending={"action": "schedule_notification"})
    rules_toolbox = _RulesToolbox(result={"answer": "Powiadomienie zapisane."})

    async def agent_invoker(
        _model_factory: ModelFactory,
        _tools: list[Any],
        _messages: Sequence[BaseMessage],
        _config: RunnableConfig,
        _system_prompt_suffix: str,
        _timeout_seconds: float,
        _logger: Logger,
    ) -> tuple[str, bool]:
        raise AssertionError("agent should not be called")

    service = _service(
        session_factory=session_factory,
        memory=memory,
        agent_invoker=agent_invoker,
        rules_toolbox=rules_toolbox,
    )

    answer = await service.handle(_request("Tak"))

    assert answer == "Powiadomienie zapisane."
    assert rules_toolbox.confirmed is True


async def test_direct_confirmation_returns_dict_tool_error() -> None:
    session_factory = _SessionFactory()
    memory = _Memory(pending={"action": "schedule_notification"})
    rules_toolbox = _RulesToolbox(result={"error": "Nie udało się zapisać powiadomienia."})

    async def agent_invoker(
        _model_factory: ModelFactory,
        _tools: list[Any],
        _messages: Sequence[BaseMessage],
        _config: RunnableConfig,
        _system_prompt_suffix: str,
        _timeout_seconds: float,
        _logger: Logger,
    ) -> tuple[str, bool]:
        raise AssertionError("agent should not be called")

    service = _service(
        session_factory=session_factory,
        memory=memory,
        agent_invoker=agent_invoker,
        rules_toolbox=rules_toolbox,
    )

    answer = await service.handle(_request("Tak"))

    assert answer == "Nie udało się zapisać powiadomienia."


async def test_normal_message_loads_history_invokes_agent_and_saves_turn() -> None:
    session_factory = _SessionFactory()
    memory = _Memory(
        turns=[
            {"role": "user", "text": "Jaka była pogoda?"},
            {"role": "bot", "text": "Było 12 C."},
        ]
    )
    captured: dict[str, Any] = {}

    async def agent_invoker(
        model_factory: ModelFactory,
        tools: list[Any],
        messages: Sequence[BaseMessage],
        config: RunnableConfig,
        system_prompt_suffix: str,
        timeout_seconds: float,
        _logger: Logger,
    ) -> tuple[str, bool]:
        captured["model_factory"] = model_factory
        captured["tools"] = tools
        captured["messages"] = messages
        captured["config"] = config
        captured["system_prompt_suffix"] = system_prompt_suffix
        captured["timeout_seconds"] = timeout_seconds
        return "Jutro będzie 14 C.", False

    service = ConversationService(
        session_factory=session_factory,  # type: ignore[arg-type]
        forecast_provider=object(),
        observation_provider=object(),
        geocoder=object(),
        model_factory=_ModelFactory(),  # type: ignore[arg-type]
        rule_expression_evaluator=object(),
        timeout_seconds=3,
        memory_factory=lambda _session: memory,
        auth_resolver=_auth_resolver,  # type: ignore[arg-type]
        weather_toolbox_factory=lambda **_kwargs: _Toolbox(["weather-tool"]),
        rules_toolbox_factory=lambda **_kwargs: _RulesToolbox(),
        agent_invoker=agent_invoker,
    )

    answer = await service.handle(_request("A jutro?"))

    assert answer == "Jutro będzie 14 C."
    assert captured["tools"] == ["weather-tool", "rules-tool"]
    assert [message.content for message in captured["messages"]] == [
        "Jaka była pogoda?",
        "Było 12 C.",
        "A jutro?",
    ]
    assert captured["config"]["configurable"] == {"thread_id": "100:7"}
    assert captured["timeout_seconds"] == 3
    assert "Europe/Warsaw" in captured["system_prompt_suffix"]
    assert [turn["role"] for _key, turn in memory.saved] == ["user", "bot"]


async def test_handle_reply_returns_tool_attachments_while_handle_returns_text() -> None:
    session_factory = _SessionFactory()
    memory = _Memory()
    attachment = BotAttachment(
        filename="prognoza.png",
        media_type="image/png",
        data=b"png",
    )

    async def agent_invoker(
        _model_factory: ModelFactory,
        _tools: list[Any],
        _messages: Sequence[BaseMessage],
        _config: RunnableConfig,
        _system_prompt_suffix: str,
        _timeout_seconds: float,
        _logger: Logger,
    ) -> tuple[str, bool]:
        return "Dołączam wykres.", False

    def weather_toolbox_factory(**kwargs: Any) -> _Toolbox:
        kwargs["reply_attachments"].append(attachment)
        return _Toolbox(["weather-tool"])

    service = ConversationService(
        session_factory=session_factory,  # type: ignore[arg-type]
        forecast_provider=object(),
        observation_provider=object(),
        geocoder=object(),
        model_factory=_ModelFactory(),  # type: ignore[arg-type]
        rule_expression_evaluator=object(),
        timeout_seconds=3,
        memory_factory=lambda _session: memory,
        auth_resolver=_auth_resolver,  # type: ignore[arg-type]
        weather_toolbox_factory=weather_toolbox_factory,
        rules_toolbox_factory=lambda **_kwargs: _RulesToolbox(),
        agent_invoker=agent_invoker,
    )

    reply = await service.handle_reply(_request("Pokaż wykres wiatru"))
    text = await service.handle(_request("Pokaż wykres wiatru"))

    assert reply.text == "Dołączam wykres."
    assert reply.attachments == (attachment,)
    assert text == "Dołączam wykres."


async def test_follow_up_message_uses_previous_persisted_turns() -> None:
    session_factory = _SessionFactory()
    memory = _KeyedMemory()
    captured_messages: list[list[str]] = []

    async def agent_invoker(
        _model_factory: ModelFactory,
        _tools: list[Any],
        messages: Sequence[BaseMessage],
        _config: RunnableConfig,
        _system_prompt_suffix: str,
        _timeout_seconds: float,
        _logger: Logger,
    ) -> tuple[str, bool]:
        captured_messages.append([str(message.content) for message in messages])
        if len(captured_messages) == 1:
            return "W Warszawie jutro będzie 18 C i słaby wiatr.", False
        return "Wiatr nadal będzie słaby, około 3 m/s.", False

    service = _service(
        session_factory=session_factory,
        memory=memory,
        agent_invoker=agent_invoker,
    )

    await service.handle(_request("Jaka będzie jutro pogoda w Warszawie?"))
    answer = await service.handle(_request("A wiatr?"))

    assert answer == "Wiatr nadal będzie słaby, około 3 m/s."
    assert captured_messages == [
        ["Jaka będzie jutro pogoda w Warszawie?"],
        [
            "Jaka będzie jutro pogoda w Warszawie?",
            "W Warszawie jutro będzie 18 C i słaby wiatr.",
            "A wiatr?",
        ],
    ]


async def test_conversation_context_is_scoped_by_thread() -> None:
    session_factory = _SessionFactory()
    memory = _KeyedMemory()
    captured: dict[str, list[str]] = {}

    async def agent_invoker(
        _model_factory: ModelFactory,
        _tools: list[Any],
        messages: Sequence[BaseMessage],
        config: RunnableConfig,
        _system_prompt_suffix: str,
        _timeout_seconds: float,
        _logger: Logger,
    ) -> tuple[str, bool]:
        thread_id = str(config["configurable"]["thread_id"])
        captured[thread_id] = [str(message.content) for message in messages]
        return f"odpowiedź dla {thread_id}", False

    service = _service(
        session_factory=session_factory,
        memory=memory,
        agent_invoker=agent_invoker,
    )

    await service.handle(_request("Pogoda w Warszawie?"))
    await service.handle(
        UserMessage(
            telegram_user_id=42,
            chat_id=100,
            message_thread_id=8,
            text="A jutro?",
            message_id=11,
            reply_to_message_id=None,
        )
    )

    assert captured["100:7"] == ["Pogoda w Warszawie?"]
    assert captured["100:8"] == ["A jutro?"]
