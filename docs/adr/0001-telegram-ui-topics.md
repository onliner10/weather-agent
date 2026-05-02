# ADR 0001: Telegram as the primary UI with topic-scoped contexts

## Context

The MVP needs a simple user interface that already supports direct user communication, notifications, and lightweight conversational threading.

## Decision

Use Telegram as the only interaction channel for the MVP, with a private supergroup plus Topics as the preferred operating mode. Conversation state will key off `chat_id + message_thread_id`, with `chat_id` fallback when no topic exists.

## Consequences

- The initial UX is optimized for chat-driven weather questions and notifications instead of a web UI.
- Thread scoping becomes a first-class contract for persistence and explicit conversation history.
- Future channels must adapt to the same domain contracts rather than reshaping them.
