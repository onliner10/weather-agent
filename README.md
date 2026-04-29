# Weather Agent

Telegram weather AI agent foundations for a Polish-language MVP built around deterministic weather and rule workflows.

## Quick start (Docker Compose)

### Prerequisites

- [Docker](https://docs.docker.com/get-docker/) and [Docker Compose](https://docs.docker.com/compose/install/) (v2 plugin recommended)

### Setup

1. Copy the environment template:

   ```bash
   cp .env.example .env
   ```

2. Set required values in `.env`:

   - `WEATHER_AGENT_TELEGRAM__BOT_TOKEN` — your Telegram bot token from [@BotFather](https://t.me/BotFather)
   - `POSTGRES_PASSWORD` — set a local database password before starting Compose
   - `WEATHER_AGENT_DATABASE_URL` — leave the template host value for local development; Docker Compose overrides it from `POSTGRES_*`

3. Start the stack:

   ```bash
   docker compose up -d
   ```

4. Verify all services are running:

   ```bash
   docker compose ps
   ```

   All three services (`weather-agent-bot`, `weather-agent-worker`, `postgres-timescaledb`) should show status `Up` (or `healthy`).

5. Check logs:

   ```bash
   docker compose logs -f
   ```

### Homelab / production notes

- Services use `restart: unless-stopped` so they survive host reboots.
- Postgres data is stored in a named Docker volume (`pgdata`). A `pgbackups` volume is mounted at `/backups` for backup scripts.
- Structured JSON logs are emitted via the `json-file` driver with 10 MB rotation and 3-file retention per service. Use `docker compose logs -f <service>` to follow.
- No secrets are baked into images — all configuration comes from the `.env` file.
- `.dockerignore` excludes `.env`, local caches, generated Beads runtime data, and common key/certificate formats from Docker build context.
- Healthchecks are wired for bot and worker (Python import check) and Postgres (`pg_isready`).

### Telegram setup

1. Create a new bot via [@BotFather](https://t.me/BotFather) and copy the token.
2. Create a **private supergroup** in Telegram.
3. Enable **Topics** in the supergroup settings.
4. Add the bot to the supergroup.
5. Set `WEATHER_AGENT_TELEGRAM__ALLOWED_USER_IDS` in `.env` to the comma-separated Telegram user IDs that should have access.

## Development (local)

1. Install [uv](https://docs.astral.sh/uv/).
2. Sync the environment: `uv sync`.
3. Copy `.env.example` to `.env` and replace placeholder values.
4. Run the bot locally: `./scripts/dev/run.sh` or `python -m weather_agent bot`.
5. Run quality gates:

   - `uv run pytest`
   - `uv run ruff check .`
   - `uv run mypy`

6. Install local git hooks that block likely secrets before commit and push:

   ```bash
   ./scripts/security/install-git-hooks.sh
   ```

   The hooks run `scripts/security/scan-secrets.sh`. If `gitleaks` is installed locally, the same hooks also run `gitleaks protect --staged` on commit and `gitleaks detect` on push.

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

See [AGENTS.md](AGENTS.md), [docs/domain-vocabulary.md](docs/domain-vocabulary.md), and [docs/adr/](docs/adr/) before starting follow-on tasks.

## Environment variables

See `.env.example` for the full list. Key variables:

| Variable | Required | Default | Description |
|---|---|---|---|
| `WEATHER_AGENT_TELEGRAM__BOT_TOKEN` | Yes | — | Telegram bot token |
| `WEATHER_AGENT_TELEGRAM__ALLOWED_USER_IDS` | Yes | — | Comma-separated authorized Telegram user IDs |
| `POSTGRES_DB` | Yes for Compose | `weather_agent` | Compose Postgres database name |
| `POSTGRES_USER` | Yes for Compose | `weather_agent` | Compose Postgres user |
| `POSTGRES_PASSWORD` | Yes for Compose | — | Compose Postgres password; keep only in local `.env` |
| `WEATHER_AGENT_DATABASE_URL` | Yes | — | PostgreSQL connection string |
| `WEATHER_AGENT_MODEL__PROVIDER` | No | `openai` | LLM provider: openai, anthropic, deepseek, glm |
| `WEATHER_AGENT_MODEL__MODEL_NAME` | No | `gpt-5-mini` | Model name for selected provider |
| `WEATHER_AGENT_MODEL__TEMPERATURE` | No | `0.2` | Model temperature |
| `WEATHER_AGENT_MODEL__API_KEY` | No | — | API key for the selected LLM provider |
| `WEATHER_AGENT_MODEL__BASE_URL` | No | — | Custom base URL (for DeepSeek, GLM, proxies) |
| `WEATHER_AGENT_OPEN_METEO__BASE_URL` | No | `https://api.open-meteo.com/v1/forecast` | Open-Meteo forecast API |
| `WEATHER_AGENT_LANGSMITH__ENABLED` | No | `false` | Enable LangSmith tracing |

## LangSmith tracing

Tracing is available via [LangSmith](https://smith.langchain.com) for monitoring conversational turns, worker evaluation cycles, and provider calls.

### Settings

| Variable | Required | Default | Description |
|---|---|---|---|
| `WEATHER_AGENT_LANGSMITH__ENABLED` | No | `false` | Master switch for tracing |
| `WEATHER_AGENT_LANGSMITH__API_KEY` | No | — | LangSmith API key |
| `WEATHER_AGENT_LANGSMITH__PROJECT` | No | `weather-agent-dev` | LangSmith project name |
| `WEATHER_AGENT_LANGSMITH__ENDPOINT` | No | `https://api.smith.langchain.com` | LangSmith endpoint |

### Honored environment variables

The following LangSmith/LangChain legacy env vars are set by `configure_tracing()` when `ENABLED=true` and are also honoured at read time:

| Env var | Set by agent | Read by LangSmith |
|---|---|---|
| `LANGCHAIN_TRACING_V2` / `LANGSMITH_TRACING_V2` | Yes (to `"true"`) | Yes |
| `LANGCHAIN_API_KEY` / `LANGSMITH_API_KEY` | Yes (from settings) | Yes |
| `LANGCHAIN_PROJECT` / `LANGSMITH_PROJECT` | Yes (from settings) | Yes |
| `LANGCHAIN_ENDPOINT` / `LANGSMITH_ENDPOINT` | Yes (from settings) | Yes |
| `LANGSMITH_TRACING` / `LANGCHAIN_TRACING` | No | Yes |

If you set any of these manually before the app starts, they take precedence. When `ENABLED=false` all of the above are removed from the environment.

### Trace hierarchy

A typical conversational turn produces this hierarchy in LangSmith:

```
telegram-turn:<context_key>:<intent>         ← top-level, run_type="chain"
├── load_thread_context                       ← chain
├── classify_intent                           ← chain
├── handle_<intent>                           ← chain
│   ├── resolve_location_node                 ← implicit via subtraces
│   ├── propose_cel_rule_llm                  ← llm (rule flow only)
│   └── weather_agent_node                    ← chain, calls tool functions
│       ├── get_forecast / get_observations   ← @traceable(run_type="tool")
│       └── resolve_location                  ← @traceable(run_type="tool")
└── save_thread_context                       ← chain
```

Worker cycles create a separate trace tree:

```
worker_cycle                          ← tool
└── evaluate_rules                    ← tool
    └── evaluate_single_rule          ← tool
        ├── forecast_refresh          ← tool
        └── evaluation_result         ← tool (sync)
```

Notification sending (triggered by the bot process) appears as:

```
send_notification                     ← chain (from sender.py)
```

### Intentionally excluded from traces

- **Raw message bodies** — `user_message` is truncated to 80 characters (`user_message_preview`). Full message text never appears in LangSmith metadata.
- **`recent_context` (turn history)** — not included in metadata to avoid leaking prior conversation content.
- **`reply_anchor`** — the replied-to message content is not included.
- **`forecast_result` / `observation_result`** — raw forecast/observation payloads are excluded; only resolved summaries (location name, time explanation) are included.
- **`pending_confirmation`** — pending rule confirmation data is not traced.
- **`cel_expression` / `cel_validation_result`** — raw CEL expressions from state are excluded.
- **Secrets** — `api_key` values are never written to trace metadata. Only environment variables are set for the LangSmith client.

### Troubleshooting

If `/status` reports LangSmith as enabled but no traces appear:

1. **Check the API key** — `has_api_key` must be `true` in the status response. Without a key, the LangSmith client silently drops traces. Set `WEATHER_AGENT_LANGSMITH__API_KEY`.
2. **Verify the endpoint** — `endpoint` in the status should match `https://api.smith.langchain.com` (or your custom endpoint). Corporate proxies or DNS issues can prevent traces from being uploaded.
3. **Check the project** — `project` in the status must exist in your LangSmith workspace. If the project name is wrong, traces are created in the default project or dropped.
4. **Confirm the trace actually ran** — the health endpoint at `/health` returns `{"langsmith_enabled": true/false}`. If it says `false`, the env vars were not set during startup.
5. **Inspect the client-side filtering** — LangSmith SDK caches env var state at import time. Restart the process after changing env vars. The `configure_tracing()` call clears the SDK cache via `get_env_var.cache_clear()`.
6. **Network connectivity** — the bot/worker processes must be able to reach `api.smith.langchain.com`. Check firewall rules and proxy settings.

## Location examples

Add locations using the `/dodaj_lok` command or through conversation:

- **Home**: `/dodaj_lok Dom 52.2297 21.0122`
- **Chwarzno**: `/dodaj_lok Chwarzno 54.4871 18.4202`
- **Jeziorak**: `/dodaj_lok Jeziorak 53.6108 19.6603`

## Rule examples

Rules are created through natural Polish conversation with the bot. The LLM proposes a CEL expression, which you confirm before activation.

| User request | Proposed CEL expression |
|---|---|
| Weekend summary | (on-demand, no rule needed) |
| Rain alert | `duration_where(precipitation_mm > 0.2, next_hours(12)) >= minutes(60)` |
| Wind gust alert | `max("wind_gusts_10m_ms", weekend()) >= 12` |
| Forecast deterioration | `forecast_delta("apparent_temperature_c", tomorrow(), previous_snapshot()) <= -7` |
| Weekly check | (schedule-based, no CEL condition needed) |

## Docker Compose deployment

```bash
# Production
docker compose up -d

# Development (with source mount hot-reload)
docker compose -f docker-compose.yml -f docker-compose.override.yml up -d

# Backups
docker compose exec postgres-timescaledb pg_dump -U weather_agent weather_agent > backup.sql

# Restore
cat backup.sql | docker compose exec -T postgres-timescaledb psql -U weather_agent weather_agent
```

## Troubleshooting

| Issue | Solution |
|---|---|
| Bot not responding | Check `WEATHER_AGENT_TELEGRAM__BOT_TOKEN` and `WEATHER_AGENT_TELEGRAM__ALLOWED_USER_IDS` in `.env` |
| Database connection failed | Verify `WEATHER_AGENT_DATABASE_URL` and that Postgres is running: `docker compose ps` |
| No weather data | Check Open-Meteo API availability and `WEATHER_AGENT_OPEN_METEO__BASE_URL` |
| Wrong timezone | Ensure `WEATHER_AGENT_DEFAULT_TIMEZONE=Europe/Warsaw` |
| LangSmith traces missing | Check `/status` for `has_api_key` and `project`. See [LangSmith tracing](#langsmith-tracing) troubleshooting. |

## Testing

```bash
# Unit and integration tests (excludes real API calls)
uv run pytest

# Smoke tests (call real external APIs, skipped by default)
uv run pytest -m smoke

# Evaluate intent and CEL generation
uv run pytest tests/eval/
```
