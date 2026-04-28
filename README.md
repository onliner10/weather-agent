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
   - `WEATHER_AGENT_DATABASE_URL` — leave the default for Docker Compose (it is overridden in `docker-compose.yml`)

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
| `WEATHER_AGENT_DATABASE_URL` | Yes | — | PostgreSQL connection string |
| `WEATHER_AGENT_MODEL__PROVIDER` | No | `openai` | LLM provider: openai, anthropic, deepseek, glm |
| `WEATHER_AGENT_MODEL__MODEL_NAME` | No | `gpt-5-mini` | Model name for selected provider |
| `WEATHER_AGENT_MODEL__TEMPERATURE` | No | `0.2` | Model temperature |
| `WEATHER_AGENT_MODEL__API_KEY` | No | — | API key for the selected LLM provider |
| `WEATHER_AGENT_MODEL__BASE_URL` | No | — | Custom base URL (for DeepSeek, GLM, proxies) |
| `WEATHER_AGENT_OPEN_METEO__BASE_URL` | No | `https://api.open-meteo.com/v1/forecast` | Open-Meteo forecast API |
| `WEATHER_AGENT_LANGSMITH__ENABLED` | No | `false` | Enable LangSmith tracing |

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
| LangSmith traces missing | Set `WEATHER_AGENT_LANGSMITH__ENABLED=true` and `WEATHER_AGENT_LANGSMITH__API_KEY` |

## Testing

```bash
# Unit and integration tests (excludes real API calls)
uv run pytest

# Smoke tests (call real external APIs, skipped by default)
uv run pytest -m smoke

# Evaluate intent and CEL generation
uv run pytest tests/eval/
```