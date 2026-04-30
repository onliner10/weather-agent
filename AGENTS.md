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
- Keep bot runtime instructions in `src/weather_agent/llm/prompts/weather_agent.md`, not in this file.

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
- Keep deterministic business logic independent from Telegram, databases, HTTP APIs, schedulers, LangChain, and LLM providers.
- Treat LLM outputs as untrusted input. Parse and validate them into typed domain values before use.
- Keep runtime rule evaluation deterministic. LLMs may propose CEL rules, but they must not execute or decide rule results.

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

## Error Handling And `returns`

Use `returns` when it makes failure, absence, dependency access, async work, or IO explicit and type-checked. Keep usage simple and readable.

- Expected failure: use `Result[T, DomainError]` with domain-specific error types.
- Optional value flowing through transformations: use `Maybe[T]`.
- Sync IO that can fail: use `IOResult[T, Error]`.
- Async IO that can fail: use `FutureResult[T, Error]`.
- Injected read-only dependencies across several functions: use `RequiresContext` or `RequiresContextResult` with a small `Protocol`.
- Pure code that cannot fail: use a plain typed return value, not a container.
- Do not call `.unwrap()`, `.failure()`, or `unsafe_perform_io` in domain or application decision code. Unwrap only in tests, outer adapters, or tiny integration glue.
- When introducing `returns` imports in production code, ensure the dependency and `returns.contrib.mypy.returns_plugin` configuration are present in the same change.

## Python Style

- Follow existing repository patterns before introducing new abstractions.
- Prefer simple functions over classes unless state, polymorphism, or a stable interface justifies a class.
- Prefer `pathlib.Path` for filesystem paths in application code.
- Use descriptive names and keep functions small enough to understand locally.
- Add comments only for non-obvious reasoning or constraints.
- Catch specific exceptions and preserve useful error context.
- Do not use `eval`, `exec`, or `pickle` with user-controlled input.
- When unsure between two valid approaches, choose the smaller change. When unsure whether a change is breaking, ask before editing.

## Current Library Documentation

- Use Context7 before relying on remembered APIs for third-party Python libraries, frameworks, SDKs, CLIs, or cloud services.
- This is especially important for LangChain, Pydantic, SQLAlchemy, FastAPI, Celery, pytest, Ruff, Mypy, Telegram libraries, and `returns`.
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
