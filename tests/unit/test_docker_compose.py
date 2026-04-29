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


def test_grafana_alloy_service_exists() -> None:
    data = _load_compose()
    assert "grafana-alloy" in data["services"]


def test_grafana_alloy_image() -> None:
    data = _load_compose()
    assert "grafana/alloy" in data["services"]["grafana-alloy"]["image"]


def test_grafana_alloy_config_volume() -> None:
    data = _load_compose()
    volumes = data["services"]["grafana-alloy"].get("volumes", [])
    volume_strs = [v if isinstance(v, str) else v.get("source", "") for v in volumes]
    assert any("ops/alloy/config.alloy" in str(v) for v in volume_strs)


def test_grafana_alloy_has_all_env_vars() -> None:
    data = _load_compose()
    env = data["services"]["grafana-alloy"].get("environment", {})
    required_vars = {
        "GRAFANA_CLOUD_PROM_URL",
        "GRAFANA_CLOUD_PROM_USER",
        "GRAFANA_CLOUD_PROM_API_KEY",
        "GRAFANA_CLOUD_LOKI_URL",
        "GRAFANA_CLOUD_LOKI_USER",
        "GRAFANA_CLOUD_LOKI_API_KEY",
    }
    assert required_vars.issubset(set(env.keys()))


def test_grafana_alloy_depends_on_bot_worker_postgres() -> None:
    data = _load_compose()
    deps = data["services"]["grafana-alloy"].get("depends_on", {})
    for svc in ("weather-agent-bot", "weather-agent-worker", "postgres-timescaledb"):
        assert svc in deps


def test_postgres_exporter_service_exists() -> None:
    data = _load_compose()
    assert "postgres-exporter" in data["services"]


def test_postgres_exporter_image() -> None:
    data = _load_compose()
    assert "prometheuscommunity/postgres-exporter" in data["services"]["postgres-exporter"]["image"]


def test_postgres_exporter_dsn_env() -> None:
    data = _load_compose()
    env = data["services"]["postgres-exporter"].get("environment", {})
    assert "DATA_SOURCE_NAME" in env
    dsn = env["DATA_SOURCE_NAME"]
    assert "postgres-timescaledb" in dsn
    assert "${POSTGRES_USER" in dsn
    assert "${POSTGRES_PASSWORD" in dsn


def test_postgres_exporter_depends_on_postgres() -> None:
    data = _load_compose()
    deps = data["services"]["postgres-exporter"].get("depends_on", {})
    assert "postgres-timescaledb" in deps


def test_new_services_on_internal_network() -> None:
    data = _load_compose()
    for svc in ("grafana-alloy", "postgres-exporter"):
        networks = data["services"][svc].get("networks", [])
        assert "internal" in networks


def test_internal_network_defined() -> None:
    data = _load_compose()
    networks = data.get("networks", {})
    assert "internal" in networks


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


def test_postgres_has_backup_volume() -> None:
    data = _load_compose()
    volumes = data["services"]["postgres-timescaledb"].get("volumes", [])
    volume_refs = [v if isinstance(v, str) else v.get("source", "") for v in volumes]
    assert any("pgbackups" in str(v) for v in volume_refs)
    assert "pgbackups" in data.get("volumes", {})


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


def test_worker_has_env_file() -> None:
    data = _load_compose()
    env_file = data["services"]["weather-agent-worker"].get("env_file", [])
    assert ".env" in env_file


def test_worker_has_database_url_override() -> None:
    data = _load_compose()
    env = data["services"]["weather-agent-worker"].get("environment", {})
    assert "WEATHER_AGENT_DATABASE_URL" in env
    assert "postgres-timescaledb" in env["WEATHER_AGENT_DATABASE_URL"]


def test_postgres_env_matches_dotenv_example() -> None:
    data = _load_compose()
    pg_env = data["services"]["postgres-timescaledb"]["environment"]
    assert pg_env["POSTGRES_DB"] == "${POSTGRES_DB:?Set POSTGRES_DB}"
    assert pg_env["POSTGRES_USER"] == "${POSTGRES_USER:?Set POSTGRES_USER}"
    assert pg_env["POSTGRES_PASSWORD"] == "${POSTGRES_PASSWORD:?Set POSTGRES_PASSWORD}"


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


def test_bot_has_healthcheck() -> None:
    data = _load_compose()
    hc = data["services"]["weather-agent-bot"].get("healthcheck")
    assert hc is not None
    assert "/health" in str(hc["test"])
    assert "8080" in str(hc["test"])
    assert hc["interval"] == "30s"
    assert hc["timeout"] == "10s"
    assert hc["retries"] == 3


def test_worker_has_healthcheck() -> None:
    data = _load_compose()
    hc = data["services"]["weather-agent-worker"].get("healthcheck")
    assert hc is not None
    assert "/health" in str(hc["test"])
    assert "8081" in str(hc["test"])
    assert hc["retries"] == 3


def test_services_have_json_logging() -> None:
    data = _load_compose()
    for name in ("weather-agent-bot", "weather-agent-worker", "postgres-timescaledb"):
        logging_cfg = data["services"][name].get("logging", {})
        assert logging_cfg.get("driver") == "json-file", (
            f"Service {name} missing json-file logging driver"
        )
        options = logging_cfg.get("options", {})
        assert "max-size" in options, f"Service {name} missing max-size log option"
        assert "max-file" in options, f"Service {name} missing max-file log option"


def test_no_secrets_in_compose_environment() -> None:
    data = _load_compose()
    sensitive_keys = {"POSTGRES_PASSWORD"}
    pg_env = data["services"]["postgres-timescaledb"]["environment"]
    for svc in ("weather-agent-bot", "weather-agent-worker"):
        env = data["services"][svc].get("environment", {})
        for key in env:
            assert key not in sensitive_keys, (
                f"Service {svc} exposes secret key {key} in compose environment"
            )
    for key, value in pg_env.items():
        if key in sensitive_keys:
            assert value.startswith("${POSTGRES_PASSWORD"), (
                f"Postgres {key} must come from .env interpolation"
            )


def test_dockerfile_has_healthcheck() -> None:
    dockerfile_path = Path(__file__).resolve().parents[2] / "Dockerfile"
    content = dockerfile_path.read_text()
    assert "HEALTHCHECK" in content


def test_dockerfile_runs_as_non_root() -> None:
    dockerfile_path = Path(__file__).resolve().parents[2] / "Dockerfile"
    content = dockerfile_path.read_text()
    assert "USER appuser" in content
