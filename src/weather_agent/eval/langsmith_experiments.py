from __future__ import annotations

import os
import re

MetadataValue = bool | str | list[str]

_DEFAULT_APPLICATION = "weather-agent-preprod"


def _env(name: str) -> str | None:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return None
    return value


def _application_name() -> str:
    return _env("LANGSMITH_APPLICATION") or _env("LANGSMITH_PROJECT") or _DEFAULT_APPLICATION


def _environment_name(application: str) -> str:
    if value := _env("WEATHER_AGENT_ENVIRONMENT"):
        return value
    if value := _env("LANGSMITH_ENVIRONMENT"):
        return value
    if application.endswith("-preprod"):
        return "preprod"
    if application.endswith("-prod"):
        return "prod"
    if application.endswith("-dev"):
        return "dev"
    return "unknown"


def _run_source() -> str:
    return "github_actions" if os.environ.get("GITHUB_ACTIONS") == "true" else "local"


def _github_run_url() -> str | None:
    server_url = _env("GITHUB_SERVER_URL")
    repository = _env("GITHUB_REPOSITORY")
    run_id = _env("GITHUB_RUN_ID")
    if server_url is None or repository is None or run_id is None:
        return None
    return f"{server_url}/{repository}/actions/runs/{run_id}"


def _add_optional_env(
    metadata: dict[str, MetadataValue],
    metadata_key: str,
    env_name: str,
) -> None:
    if value := _env(env_name):
        metadata[metadata_key] = value


def build_langsmith_eval_metadata(
    *,
    eval_suite: str,
    dataset_name: str,
    git_sha: str,
) -> dict[str, MetadataValue]:
    application = _application_name()
    run_source = _run_source()
    metadata: dict[str, MetadataValue] = {
        "application": application,
        "langsmith_project": _env("LANGSMITH_PROJECT") or application,
        "environment": _environment_name(application),
        "eval_kind": "model_quality",
        "eval_suite": eval_suite,
        "dataset_name": dataset_name,
        "run_source": run_source,
        "ci": run_source == "github_actions",
        "git_sha": git_sha,
    }

    _add_optional_env(metadata, "github_workflow", "GITHUB_WORKFLOW")
    _add_optional_env(metadata, "github_run_id", "GITHUB_RUN_ID")
    _add_optional_env(metadata, "github_run_attempt", "GITHUB_RUN_ATTEMPT")
    _add_optional_env(metadata, "github_repository", "GITHUB_REPOSITORY")
    _add_optional_env(metadata, "github_ref", "GITHUB_REF")
    _add_optional_env(metadata, "github_ref_name", "GITHUB_REF_NAME")
    _add_optional_env(metadata, "github_sha", "GITHUB_SHA")
    if run_url := _github_run_url():
        metadata["github_run_url"] = run_url
    return metadata


def _slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-")
    return slug or "unknown"


def build_langsmith_eval_experiment_prefix(
    *,
    eval_suite: str,
    model_provider: str,
    model_name: str,
) -> str:
    application = _slug(_application_name())
    suite = _slug(eval_suite)
    provider = _slug(model_provider)
    model = _slug(model_name)
    return f"{application}--ci-eval--{suite}--{provider}--{model}"
