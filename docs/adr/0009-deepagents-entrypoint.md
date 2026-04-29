# ADR 0009: DeepAgents agent entrypoint for conversation runtime

## Status

Accepted

## Context

The project needs a maintainable runtime for Telegram weather conversations that handles tool execution, message history, thread persistence, and instruction management.

The current architecture uses an imperative `ConversationService.handle_turn()` with `if/elif` intent routing and a hand-written weather tool loop in `weather_handler.py` that builds raw OpenAI-style tool schemas and manually dispatches `tool_calls`. A `CompiledConversationService` shim mimics LangGraph's `ainvoke()` interface but is not a real graph.

The DeepAgents framework (`create_deep_agent`) provides a ready-made ReAct agent with planning, filesystem, subagent capabilities, and native LangGraph checkpointing. Reference examples at `/tmp/deepagents/examples` (text-to-sql, deep_research, content-builder) show small code entrypoints that compose model, tools, memory, skills, and subagents.

A minimal `StateGraph` with `ToolNode` and manual edges would also work, but DeepAgents provides more built-in capabilities (planning via `write_todos`, progressive skill loading, subagent delegation) that future features may need, and its usage aligns with the LangGraph learning-stack decision in ADR 0002.

## Decision

Use `create_deep_agent(...)` as the sole agent entrypoint for the active conversation runtime.

### Entrypoint

A small `create_weather_agent()` factory in `src/weather_agent/agent_factory.py` composes:

```python
create_deep_agent(
    model=chat_model,
    memory=["./AGENTS.md"],
    tools=weather_tools,
    subagents=[],
)
```

- `memory=["./AGENTS.md"]` loads assistant identity, language constraints, safety rules, and workflow instructions as a static system prompt fragment.
- `skills=["./skills/"]` is deferred — the MVP AGENTS.md is sufficient. Skills can be added later when workflows (weather Q&A vs rule proposal) grow complex enough to need separate files.
- `subagents=[]` for MVP — the weather assistant is single-agent. Future features (multi-location comparison, deep research) may add subagents.
- No `system_prompt` parameter is used; all instructions come from `memory=["./AGENTS.md"]` as the reference examples prescribe.

### Thread identity

Telegram topic-aware context keys (`chat_id:message_thread_id` with `chat_id` fallback) map directly to DeepAgents/LangGraph thread config:

```python
agent.ainvoke(
    {"messages": [("user", user_message)]},
    config={"configurable": {"thread_id": context_key}},
)
```

### Backend / checkpointing

`deepagents` uses LangGraph's `MemorySaver` by default (in-memory). For MVP the existing `ThreadMemoryService` + `TelegramContextService` (Postgres-backed) continues to handle:
- Turn persistence (last 20 turns with TTL)
- Pending confirmation metadata
- Reply context across restarts

A DeepAgents/LangGraph checkpointer (e.g., `PostgresSaver`) should be evaluated post-MVP for full checkpointed state including agent internal state across restarts.

### Instructions

- `AGENTS.md` at project root holds durable assistant instructions: Polish-only constraint, tool descriptions, safety rules (LLM may propose CEL rules but deterministic validation and persistence is required), and workflow guidance.
- Prompt modules remain for small dynamic fragments: date/time rendering, CEL allowlist rendering.
- `skills/` directory structure is created for future use but not actively loaded in the MVP factory.

### Tool wiring

Existing `WeatherToolbox.to_langchain_tools()` produces `StructuredTool` objects. These are passed directly to `create_deep_agent(tools=...)`. The hand-written tool dispatch loop in `weather_handler.py` is replaced by DeepAgents' built-in ReAct tool execution.

### Layers to retire

1. `CompiledConversationService` shim — replaced by direct agent invocation
2. `graphs/conversation.py` compatibility wrapper — no longer imported by active paths
3. `graphs/nodes/weather_qa.py` legacy node — duplicated by weather_handler.py
4. `graphs/nodes/rule_management.py` legacy node — duplicated by rule_handler.py
5. `graphs/state.py` — replaced by DeepAgents' internal `messages` state
6. Manual tool dispatch loop in `weather_handler.py.handle_weather()`

### Layers to keep

1. `application/rules/rule_handler.py` — rule proposal, confirmation, cancel logic (deterministic validation/persistence path)
2. `application/context_service.py` — turn persistence, reply context (until checkpointing is in place)
3. `infrastructure/memory/thread_memory.py` — Postgres-backed turn history
4. `adapters/telegram/context.py` — topic-aware context key management
5. `llm/tools/weather_tools.py` + `WeatherToolbox` — LangChain tool implementations
6. `domain/` — all domain models, providers, CEL evaluator

### Intent classification

Intent classification remains as-is for the non-DeepAgents path (rule proposal CEL validation, pending confirmation routing). With DeepAgents, the agent natively handles tool selection — explicit intent routing is only needed for domain-level branching (weather vs rule vs command).

## Consequences

- The active Telegram runtime imports `create_weather_agent()` instead of `compile_conversation_service()`.
- Tool dispatch is handled by DeepAgents/LangGraph's internal ReAct loop.
- Conversation state is message-centric (`{"messages": [...]}`) internally.
- Intent routing is simplified: DeepAgents handles tool selection; the app layer only routes weather vs rule vs command.
- Legacy graph-shaped code can be deleted once the DeepAgents path is tested.
- DeepAgents' `write_todos` planning and subagent delegation are available for future features without code changes.
- Runtime rule validation/evaluation remains deterministic and never calls the LLM.
