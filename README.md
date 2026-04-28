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