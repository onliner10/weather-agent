# ADR 0002: LangChain tools and LangSmith for observable agent runtime

## Context

The project uses an LLM for Polish-language conversation, but correctness must stay anchored in deterministic weather providers, database state, and rule evaluation.

Earlier versions targeted heavier graph-style orchestration. That was more machinery than the current runtime needs: there are no subagents, no planner hierarchy, and database conversation history is already the source of truth.

## Decision

Use a small LangChain tool-calling runtime and LangSmith for trace and evaluation visibility.

The runtime shape is:

```text
Telegram adapter -> ConversationService -> AgentRuntime -> deterministic tools -> DB/providers
```

Conversation state is loaded explicitly from the database and passed to the model as messages. Do not add a separate checkpointer or in-memory saver unless there is a concrete need that database history cannot satisfy.

## Consequences

- Tool calls and conversation turns can be inspected without turning the system into an opaque agent.
- Deterministic services remain testable outside the orchestration stack.
- Runtime rule execution must continue to work even if tracing or LLM tooling is unavailable.
- Additional orchestration libraries should be added only when they remove current complexity, not for speculative future workflows.
