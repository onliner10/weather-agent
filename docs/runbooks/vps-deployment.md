# VPS Deployment

This project deploys to a VPS with Docker Compose. The VPS runs the bot and worker
containers. PostgreSQL is expected to be an external hosted database.

## One-Time VPS Setup

1. Install Docker and either the Docker Compose plugin (`docker compose`) or legacy
   `docker-compose` on the VPS.
2. Create the deployment directory:

   ```bash
   sudo mkdir -p /opt/weather-agent
   sudo chown "$USER:$USER" /opt/weather-agent
   ```

3. Create `/opt/weather-agent/.env` on the VPS. Do not commit this file.

   ```env
   WEATHER_AGENT_TELEGRAM__BOT_TOKEN=...
   WEATHER_AGENT_TELEGRAM__ALLOWED_USER_IDS=123456789
   WEATHER_AGENT_DATABASE_URL=postgresql+psycopg://user:password@host:5432/dbname?sslmode=require
   WEATHER_AGENT_MODEL__PROVIDER=openai
   WEATHER_AGENT_MODEL__MODEL_NAME=gpt-4.1-mini
   WEATHER_AGENT_MODEL__API_KEY=...
   WEATHER_AGENT_LANGSMITH__ENABLED=false
   WEATHER_AGENT_OPEN_METEO__BASE_URL=https://api.open-meteo.com/v1/forecast
   WEATHER_AGENT_OPEN_METEO__MODEL=dwd-icon
   WEATHER_AGENT_OPEN_METEO__TIMEOUT_SECONDS=15
   WEATHER_AGENT_DEFAULT_TIMEZONE=Europe/Warsaw
   WEATHER_AGENT_DEFAULT_LANGUAGE=pl-PL
   ```

5. Ensure the hosted PostgreSQL database exists and accepts connections from the VPS.

## GitHub Configuration

Create a `production` environment in GitHub and add these secrets:

| Secret | Description |
|---|---|
| `VPS_HOST` | VPS hostname or IP |
| `VPS_USER` | SSH user used for deployment |
| `VPS_PORT` | SSH port, usually `22` |
| `VPS_SSH_PRIVATE_KEY` | Private key for the deployment user |

Runtime secrets such as Telegram, model provider, PostgreSQL, LangSmith, and Grafana
credentials should stay in `/opt/weather-agent/.env` on the VPS.

## Deployment

Every push to `main` runs quality checks, builds an image, pushes it to GHCR, uploads
`docker-compose.prod.yml`, writes `/opt/weather-agent/.deploy.env` with the selected
image tag, runs Alembic migrations on the VPS, and restarts the bot and worker.

You can also deploy manually from GitHub Actions using the `Deploy` workflow's
`workflow_dispatch` trigger.

## Manual Operations

From the VPS:

```bash
cd /opt/weather-agent
docker compose --env-file .deploy.env -f docker-compose.prod.yml ps
docker compose --env-file .deploy.env -f docker-compose.prod.yml logs -f weather-agent-bot
docker compose --env-file .deploy.env -f docker-compose.prod.yml logs -f weather-agent-worker
```

To restart after changing `/opt/weather-agent/.env`:

```bash
cd /opt/weather-agent
docker compose --env-file .deploy.env -f docker-compose.prod.yml up -d
```
