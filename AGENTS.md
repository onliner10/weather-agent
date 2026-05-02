# Weather Agent Repository Instructions

Use these instructions as operational rules for this repository. Keep changes small, typed, tested, and aligned with the local architecture.

## Hard Rules

- Use Beads for implementation task tracking before writing code.
- Inspect the request, relevant files, and existing patterns before editing.
- Make the smallest focused change that solves the task. Do not refactor unrelated code.
- Preserve public interfaces unless the user explicitly asks for a breaking change.
- Add or update tests for behavior changes.
- Run relevant checks before handing work back.
- Do not commit or push unless the user explicitly asks. Never run `git push` as an automatic session-close step.
- Keep user-facing weather and notification flows in Polish.
- Keep bot runtime instructions in `src/weather_agent/llm/prompts/weather_agent.md`, not in this file. `AGENTS.md` is an OpenCode/system prompt for coding agents only; never load it from application runtime code or include it in LLM prompts sent by the bot.

## Default Workflow

1. Read the request and inspect relevant files.
2. If code will change, create or update a Beads issue and mark it in progress.
3. Make the smallest focused change.
4. Add or update deterministic tests when behavior changes.
5. Run the narrowest relevant checks first, then broader checks when appropriate.
6. Summarize changed files, checks run, and remaining risks.
7. Close the Beads issue when the task is finished.
8. Do not commit or push unless explicitly requested.

## Project Boundaries

- Keep repository instructions, development workflow, and coding conventions in `AGENTS.md`.
- Do not treat `AGENTS.md` as product data, runtime configuration, model context, or application prompt text. If runtime prompt content is needed, use files under `src/weather_agent/llm/prompts/` or explicit configuration.
- Keep deterministic business logic independent from Telegram, databases, HTTP APIs, schedulers, LangChain, and LLM providers.
- Treat LLM outputs as untrusted input. Parse and validate them into typed domain values before use.
- Keep runtime rule evaluation deterministic. LLMs may propose rule expressions, but they must not execute or decide rule results.

## Implementation Pitfalls

These rules address common mistakes in this repository. Apply them before choosing an implementation.

- Async SQLAlchemy sessions are not concurrency-safe. If parallel tool calls can touch the same `AsyncSession`, protect all session-using tools with one shared lock or give each tool call its own session. Separate locks per toolbox do not protect a shared session.
- Prefer short session/transaction boundaries at the adapter or service boundary. Do not keep a failed `AsyncSession` alive without `rollback()`, and do not let a rollback for one item discard earlier successful writes that the code reports as successful.
- If code flushes database changes and later code must make those changes durable, explicitly commit at the correct boundary. For example, disabling fired `once:` rules must be committed, not only flushed.
- Do not run async applications in worker threads to paper over lifecycle issues. Keep Telegram/PTB, HTTP clients, SQLAlchemy async engines, and other async resources on one event loop; use async `start()`/`stop()` APIs when available.
- Preserve shipped runtime behavior when replacing lifecycle helpers. For Telegram polling, keep `drop_pending_updates=True` unless the task explicitly asks to process backlog updates.
- When injecting a shared `httpx.AsyncClient`, preserve per-provider settings such as timeouts by passing request-level `timeout=` values or configuring the shared client equivalently. Dependency injection must not silently change provider behavior.
- Treat persisted JSON and database metadata as untrusted. Pydantic `model_validate()` can fail on old or malformed data; catch validation errors at the boundary, log useful context, clear bad state when safe, and return Polish user-facing errors in Telegram flows.

## Architecture

Do:

- Prefer hexagonal architecture: domain core, application orchestration, ports, and adapters.
- Put IO at the borders: HTTP, database, Telegram, filesystem, environment, current time, randomness, schedulers, LLMs, and external APIs.
- Keep domain logic pure where practical: same inputs produce same outputs, no hidden IO, no global state.
- Inject dependencies through explicit parameters, small protocols, or constructors.
- Separate deciding from doing: pure code decides, adapters perform side effects.

Don't:

- Call Telegram, databases, HTTP APIs, schedulers, or LLM providers from domain code.
- Reach into global settings or process state from domain logic.
- Add compatibility layers unless persisted data, shipped behavior, external consumers, or the user require them.

## Types And Data

- All functions and methods must have precise parameter and return types.
- Avoid `Any`. If it is unavoidable at an external boundary, narrow it immediately and do not let it leak into application or domain code.
- Do not pass untyped `dict` values through the codebase. Use Pydantic models, frozen dataclasses, `TypedDict`, `NamedTuple`, or explicit domain types.
- Prefer immutable value objects: `@dataclass(frozen=True)`, frozen Pydantic models, tuples, `frozenset`, and read-only interfaces such as `Sequence` or `Mapping`.
- Parse and validate external input once at the boundary, then operate on trusted typed values.
- Keep `mypy` clean under the repository's strict configuration. Suppress type errors only narrowly and at the boundary that requires it.

## Error Handling

- Use precise exceptions or explicit typed result values for expected failures.
- Prefer small domain-specific error types over stringly typed failure handling.
- Keep pure code on plain typed return values when it cannot fail.
- Do not introduce effect/container libraries unless the dependency and mypy configuration are added in the same focused change and the readability tradeoff is clearly worth it.

## Python Style

- Follow existing repository patterns before introducing new abstractions.
- Prefer simple functions over classes unless state, polymorphism, or a stable interface justifies a class.
- Prefer list comprehensions over manual append loops when they remain readable.
- Prefer `pathlib.Path` for filesystem paths in application code.
- Use descriptive names and keep functions small enough to understand locally.
- Add comments only for non-obvious reasoning or constraints.
- Catch specific exceptions and preserve useful error context.
- Do not use `eval`, `exec`, or `pickle` with user-controlled input.
- When unsure between two valid approaches, choose the smaller change. When unsure whether a change is breaking, ask before editing.

## Current Library Documentation

- Use Context7 before relying on remembered APIs for third-party Python libraries, frameworks, SDKs, CLIs, or cloud services.
- This is especially important for LangChain, Pydantic, SQLAlchemy, FastAPI, Celery, pytest, Ruff, Mypy, and Telegram libraries.
- Start by resolving the exact library name, then query the concrete API, configuration, setup, migration, or debugging question.
- If Context7 cannot identify the library or answer the question, try one alternate package name or a more specific query, then state uncertainty and fall back to local code, installed package metadata, or official docs.
- Do not use Context7 for this repository's own code, pure Python language features, business logic, architecture decisions, or deterministic domain rules unless a third-party API is directly involved.

## Testing And Quality

- Unit tests should cover pure domain logic without network, database, Telegram, or real LLM calls.
- Integration tests may cover adapters and external boundaries, but must be clearly scoped.
- Prefer deterministic tests with explicit clocks, fixed inputs, and no hidden global state.
- Prefer small fake implementations of ports over mocks when they are clearer and type-safe.
- For behavior changes, verify that tests would fail if the new logic were removed.
- Relevant checks usually include `uv run pytest`, `uv run ruff check .`, `uv run ruff format --check .`, `uv run mypy`, and `scripts/quality/check-dead-code.sh`.

## Documentation

- Document public APIs with concise Google-style docstrings when behavior is not obvious from names and types.
- Focus documentation on why a decision exists, invariants, side effects, and boundary contracts.
- Keep README and ADR updates focused on durable project knowledge, not transient implementation notes.
