"""Validate docker-compose.yml structure and required services."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

COMPOSE_PATH = Path(__file__).resolve().parents[2] / "docker-compose.yml"


def _load_compose() -> dict[str, Any]:
    data: dict[str, Any] = yaml.safe_load(COMPOSE_PATH.read_text())
    return data


def test_compose_file_is_valid_yaml() -> None:
    data = _load_compose()
    assert isinstance(data, dict)
    assert "services" in data


def test_compose_has_required_services() -> None:
    data = _load_compose()
    services = data["services"]
    required = {"weather-agent-bot", "weather-agent-worker", "postgres-timescaledb"}
    assert required.issubset(set(services.keys())), (
        f"Missing services: {required - set(services.keys())}"
    )


def test_postgres_uses_timescaledb_image() -> None:
    data = _load_compose()
    image = data["services"]["postgres-timescaledb"]["image"]
    assert "timescale/timescaledb" in image


def test_postgres_has_healthcheck() -> None:
    data = _load_compose()
    hc = data["services"]["postgres-timescaledb"].get("healthcheck")
    assert hc is not None
    assert "pg_isready" in str(hc["test"])


def test_postgres_has_persistent_volume() -> None:
    data = _load_compose()
    volumes = data["services"]["postgres-timescaledb"].get("volumes", [])
    volume_refs = [v if isinstance(v, str) else v.get("source", "") for v in volumes]
    assert any("pgdata" in str(v) for v in volume_refs)
    assert "pgdata" in data.get("volumes", {})


def test_bot_depends_on_postgres() -> None:
    data = _load_compose()
    deps = data["services"]["weather-agent-bot"].get("depends_on", {})
    assert "postgres-timescaledb" in deps


def test_worker_depends_on_postgres() -> None:
    data = _load_compose()
    deps = data["services"]["weather-agent-worker"].get("depends_on", {})
    assert "postgres-timescaledb" in deps


def test_bot_has_env_file() -> None:
    data = _load_compose()
    env_file = data["services"]["weather-agent-bot"].get("env_file", [])
    assert ".env" in env_file


def test_bot_has_database_url_override() -> None:
    data = _load_compose()
    env = data["services"]["weather-agent-bot"].get("environment", {})
    assert "WEATHER_AGENT_DATABASE_URL" in env
    assert "postgres-timescaledb" in env["WEATHER_AGENT_DATABASE_URL"]


def test_postgres_env_matches_dotenv_example() -> None:
    data = _load_compose()
    pg_env = data["services"]["postgres-timescaledb"]["environment"]
    assert pg_env["POSTGRES_DB"] == "weather_agent"
    assert pg_env["POSTGRES_USER"] == "weather_agent"
    assert pg_env["POSTGRES_PASSWORD"] == "weather_agent"


def test_no_redis_service() -> None:
    data = _load_compose()
    services = set(data["services"].keys())
    assert not any("redis" in s for s in services)


def test_no_langfuse_service() -> None:
    data = _load_compose()
    services = set(data["services"].keys())
    assert not any("langfuse" in s for s in services)


def test_services_have_restart_policy() -> None:
    data = _load_compose()
    for name in ("weather-agent-bot", "weather-agent-worker", "postgres-timescaledb"):
        assert data["services"][name].get("restart") == "unless-stopped", (
            f"Service {name} missing restart: unless-stopped"
        )