# Weather Agent

Telegram weather AI agent foundations for a Polish-language MVP built around deterministic weather and rule workflows.

## Current scope

This repository currently establishes phase 1 of the implementation plan:

- `uv`-managed Python project layout under `src/weather_agent/`
- typed environment-backed settings with safe secret handling
- baseline test and lint tooling
- architecture ADRs and domain vocabulary for fresh-session contributors

## Quick start

1. Install `uv`.
2. Sync the environment with `uv sync`.
3. Run quality gates:
   - `uv run pytest`
   - `uv run ruff check .`
   - `uv run mypy`

Copy `.env.example` to `.env` and replace placeholder values before running application code.

## Repository layout

- `src/weather_agent/` contains the importable application package.
- `tests/` contains unit and integration tests.
- `docs/adr/` captures architecture decisions.
- `docs/domain-vocabulary.md` defines stable project terminology.

## Operating constraints

- User-facing weather and rule flows are Polish-only for the MVP.
- The default timezone is `Europe/Warsaw`.
- LLMs may propose or edit rules, but runtime rule evaluation must remain deterministic and must not call the LLM.
- Beads is the only task tracker for implementation work.

See [AGENTS.md](/home/mateusz/git/weather-agent/AGENTS.md), [docs/domain-vocabulary.md](/home/mateusz/git/weather-agent/docs/domain-vocabulary.md), and [docs/adr/](/home/mateusz/git/weather-agent/docs/adr) before starting follow-on tasks.
