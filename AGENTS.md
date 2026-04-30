# Weather Agent Repository Instructions

Development style for this repository: use an explicit workflow, make small focused changes, write typed Python, prefer deterministic tests, preserve stable public interfaces, and explain concisely why a change exists. Adapt these practices to the local codebase instead of importing rules from other projects.

## Runtime Prompt

The weather bot system prompt lives in `src/weather_agent/llm/prompts/weather_agent.md` and is loaded by `weather_agent.agent_factory`.

Do not put bot persona, tool-use policy, or end-user conversational instructions in this file. Keep those in the runtime prompt file so development agents do not inherit application instructions.

## Workflow

- Use Beads for implementation task tracking before writing code.
- First read the issue or user request, inspect relevant files, and identify existing patterns before editing.
- Keep changes minimal and focused. Do not refactor unrelated code.
- Preserve public interfaces unless the user explicitly asks for a breaking change.
- Add or update tests for behavior changes.
- Run relevant quality checks before handing work back.
- Do not commit or push changes unless the user explicitly asks for it. In particular, never run `git push` as an automatic session-close step.

## Current Library Documentation

- Use the Context7 MCP before relying on remembered API details for Python libraries, frameworks, SDKs, CLIs, or cloud services. This applies to implementation, debugging, setup, migration, dependency configuration, and examples involving third-party packages such as LangChain, Pydantic, SQLAlchemy, FastAPI, Celery, pytest, Ruff, Mypy, or Telegram libraries.
- Start with Context7 library resolution for the exact package name, choose the best matching documented library by name, relevance, source reputation, available snippets, and version when the task names one, then query the docs with the concrete question or API surface being used.
- Use Context7 results to confirm current signatures, imports, configuration keys, deprecations, async behavior, typing expectations, and recommended patterns before editing code. Prefer the documented API over memory or old examples.
- If Context7 cannot identify the library or the retrieved docs do not answer the question, try one alternate package name or a more specific query. If still unresolved, state the uncertainty and fall back to local code, installed package metadata, or official docs rather than guessing.
- Do not use Context7 for this repository's own code, pure Python language features, business logic, architecture decisions, or deterministic domain rules unless a third-party library API is directly involved.

## Project Boundaries

- Keep user-facing weather and notification flows in Polish.
- Keep bot runtime instructions in `src/weather_agent/llm/prompts/weather_agent.md`.
- Keep repository instructions, development workflow, and coding conventions in `AGENTS.md`.
- Keep deterministic business logic independent from Telegram, databases, HTTP APIs, schedulers, LangChain, and LLM providers.

## Architecture Principles

- Prefer hexagonal architecture: domain core, application orchestration, ports, and adapters.
- Put IO at the borders. HTTP, database access, Telegram calls, filesystem reads, environment variables, current time, randomness, and LLM calls belong in adapters or explicit orchestration code.
- Keep domain logic pure where possible: same inputs produce same outputs, no hidden mutation, no hidden IO, no global state.
- Inject dependencies through explicit parameters, protocols, or constructors. Do not reach into globals from domain code.
- Separate deciding from doing. Pure code should decide what should happen; adapters should perform the side effects.
- Treat LLM outputs as untrusted input. Parse, validate, and convert them into typed domain values before use.
- Keep runtime rule evaluation deterministic. LLMs may propose CEL rules, but they must not execute or decide rule results.

## Immutability First

- Prefer immutable value objects over mutable dictionaries and in-place updates.
- Use `@dataclass(frozen=True)`, frozen Pydantic models, tuples, `frozenset`, and read-only interfaces such as `Sequence` or `Mapping` where practical.
- Return new values instead of mutating inputs.
- Avoid shared mutable default values. Use explicit construction at the boundary.
- Keep caches and stateful optimizations outside the domain core unless they are clearly isolated and tested.

## Type System Rules

- Use the Python type system as a design tool, not only as documentation.
- All functions and methods must have precise parameter and return types.
- Avoid `Any`. If `Any` is unavoidable at an external boundary, isolate it there, narrow it immediately, and do not let it leak into domain or application code.
- Prefer `object` for opaque values, `Protocol` for structural interfaces, and generics for reusable typed behavior.
- Do not pass untyped `dict` values through the codebase. Use Pydantic models, frozen dataclasses, `TypedDict`, `NamedTuple`, or explicit domain types.
- Model variants with `Enum`, `Literal`, discriminated unions, and `match` where it improves exhaustiveness.
- Use `NewType` or small value objects for identifiers and domain concepts when raw `str` or `int` values become ambiguous.
- Make illegal states unrepresentable where Python allows it. Parse and validate once at the boundary, then operate on trusted typed values.
- Prefer total functions. Represent expected failure with typed results, explicit exceptions, or domain-specific error types rather than sentinel values like `None` when absence is not a valid success case.
- Keep `mypy` clean under the repository's strict configuration. Do not silence type errors unless the suppression is narrow, justified, and located at the boundary that requires it.

## `returns` Library Rules

Use the `returns` library whenever it can make failure, absence, dependency access, async work, or IO explicit and type-checked. The goal is not clever functional style; the goal is code where signatures reveal what can happen.

- Use `Result[Value, Error]` for expected failures instead of returning `None`, `False`, magic strings, or raising exceptions from normal business paths.
- Use domain-specific error types in `Result`, such as frozen dataclasses, enums, or precise exception classes. Avoid `Result[T, str]`, `Result[T, Exception]`, and especially `Result[Any, Any]` in domain code unless the value is immediately narrowed at a boundary.
- Use `Maybe[T]` for optional values that flow through multiple transformations. Convert external `Optional[T]` into `Maybe[T]` once, then compose with `.map`, `.bind_optional`, or `.bind` instead of writing nested `if value is not None` blocks.
- Use `IO[T]` only to mark synchronous effects that cannot fail, such as reading the current time or randomness. Pure domain functions should not return `IO` unless they are explicitly describing an effect for an outer layer to run.
- Use `IOResult[T, Error]` for synchronous effects that can fail, such as HTTP, filesystem, database, environment variable, Telegram, scheduler, or LLM provider calls.
- Use `Future[T]` or `FutureResult[T, Error]` for async workflows. Prefer `FutureResult` for async calls to external systems so exceptions do not escape and break event-loop flows unpredictably.
- Use `RequiresContext[Value, Deps]` or `RequiresContextResult[Value, Error, Deps]` for typed functional dependency injection when passing configuration, clocks, ports, or service dependencies through several pure functions would otherwise pollute every signature.
- Define dependency inputs as small `Protocol` types. Do not pass broad containers, global settings objects, or untyped dictionaries into `RequiresContext`.
- Use `flow`, `pipe`, `.map`, `.bind`, `.alt`, and `.lash` to compose clear pipelines. Prefer readable named functions over dense point-free chains when the pipeline becomes hard to scan.
- Use `@safe`, `@impure_safe`, and `@future_safe` at adapters or boundary wrappers to convert throwing APIs into typed containers. Do not decorate arbitrary domain logic just to hide exceptions.
- Do not call `.unwrap()`, `.failure()`, or `unsafe_perform_io` in domain or application decision code. Unwrap only at the outermost adapter boundary, in tests, or in tiny integration glue where the side effect is actually executed.
- Match on `Success` and `Failure` or use container methods to handle both tracks. Never ignore the failure side of a `Result` or `IOResult`.
- When introducing `returns` imports in production code, make sure the project dependency and `returns.contrib.mypy.returns_plugin` configuration are present in the same change. Do not silently import a library that is not installed.
- Keep `returns` usage simple for future maintainers. If a plain frozen dataclass, `Enum`, or small pure function communicates the state better than a container, use the simpler typed construct.

Quick decision table for new code:

- Value may be absent and absence is valid: use `Maybe[T]`.
- Operation can fail in an expected way: use `Result[T, DomainError]`.
- Operation performs sync IO and can fail: use `IOResult[T, Error]`.
- Operation performs async IO and can fail: use `FutureResult[T, Error]`.
- Function needs injected read-only dependencies: use `RequiresContext` or `RequiresContextResult` with a small `Protocol`.
- Code is pure and cannot fail: use a plain typed return value, not a container.

## SOLID Rules

- Single Responsibility: each module, class, and function should have one reason to change.
- Open/Closed: extend behavior through new implementations of protocols or ports instead of editing stable domain logic unnecessarily.
- Liskov Substitution: implementations of a protocol must honor the contract expected by callers.
- Interface Segregation: keep ports small and role-specific. Do not create broad service interfaces that force adapters to implement unused methods.
- Dependency Inversion: domain and application layers depend on abstractions. Adapters depend on concrete frameworks and external services.

## Haskell-Inspired Python Practices

- Prefer pure functions and explicit data flow over object mutation.
- Prefer expression-oriented transformations over stepwise mutation when readability is preserved.
- Keep effectful code shallow and obvious. A reader should be able to see where IO starts.
- Parse, do not repeatedly validate. Convert external input into typed internal data before calling core logic.
- Use small composable functions with clear types instead of large procedures with implicit state.
- Avoid boolean flag parameters that create multiple behaviors. Use separate functions or typed variants.
- Use exhaustive handling for known variants. If a new variant is added, type checking and tests should guide necessary updates.

## Python Style

- Follow existing repository patterns before introducing new abstractions.
- Prefer simple functions over classes unless state, polymorphism, or a stable interface justifies a class.
- Prefer `pathlib.Path` for filesystem paths in application code.
- Use descriptive names and keep functions small enough to understand locally.
- Add comments only for non-obvious reasoning or constraints, not for restating code.
- Avoid broad `except` blocks. Catch specific exceptions and preserve useful error context.
- Do not use `eval`, `exec`, or `pickle` with user-controlled input.

## Testing And Quality

- Unit tests should cover pure domain logic without network, database, Telegram, or real LLM calls.
- Integration tests may cover adapters and external boundaries, but must be clearly scoped.
- Prefer deterministic tests with explicit clocks, fixed inputs, and no hidden global state.
- Avoid mocks when a small fake implementation of a port is clearer and type-safe.
- For behavior changes, verify that tests would fail if the new logic were removed.
- Relevant checks usually include `uv run pytest`, `uv run ruff check .`, `uv run ruff format --check .`, `uv run mypy`, and `scripts/quality/check-dead-code.sh`.

## Documentation

- Document public APIs with concise Google-style docstrings when behavior is not obvious from names and types.
- Focus documentation on why a decision exists, invariants, side effects, and boundary contracts.
- Keep README and ADR updates focused on durable project knowledge, not transient implementation notes.
