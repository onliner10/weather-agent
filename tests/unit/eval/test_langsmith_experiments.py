from __future__ import annotations

import pytest

from weather_agent.eval.langsmith_experiments import (
    build_langsmith_eval_experiment_prefix,
    build_langsmith_eval_metadata,
)


def test_langsmith_eval_metadata_marks_github_actions_runs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LANGSMITH_PROJECT", "weather-agent-preprod")
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    monkeypatch.setenv("GITHUB_WORKFLOW", "Model Quality Evals")
    monkeypatch.setenv("GITHUB_RUN_ID", "25256815013")
    monkeypatch.setenv("GITHUB_RUN_ATTEMPT", "1")
    monkeypatch.setenv("GITHUB_REPOSITORY", "onliner10/weather-agent")
    monkeypatch.setenv("GITHUB_REF", "refs/heads/master")
    monkeypatch.setenv("GITHUB_REF_NAME", "master")
    monkeypatch.setenv("GITHUB_SHA", "77e11016af535e962629c3476b64b2a401f676db")
    monkeypatch.setenv("GITHUB_SERVER_URL", "https://github.com")

    metadata = build_langsmith_eval_metadata(
        eval_suite="weather-functional",
        dataset_name="weather-agent-weather-functional-v5",
        git_sha="77e1101",
    )

    assert metadata["application"] == "weather-agent-preprod"
    assert metadata["langsmith_project"] == "weather-agent-preprod"
    assert metadata["environment"] == "preprod"
    assert metadata["eval_kind"] == "model_quality"
    assert metadata["eval_suite"] == "weather-functional"
    assert metadata["dataset_name"] == "weather-agent-weather-functional-v5"
    assert metadata["run_source"] == "github_actions"
    assert metadata["ci"] is True
    assert metadata["github_workflow"] == "Model Quality Evals"
    assert metadata["github_run_id"] == "25256815013"
    assert metadata["github_run_url"] == (
        "https://github.com/onliner10/weather-agent/actions/runs/25256815013"
    )


def test_langsmith_eval_metadata_allows_explicit_application_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LANGSMITH_PROJECT", "weather-agent-preprod")
    monkeypatch.setenv("LANGSMITH_APPLICATION", "weather-agent-preprod-evals")
    monkeypatch.setenv("LANGSMITH_ENVIRONMENT", "staging")
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)

    metadata = build_langsmith_eval_metadata(
        eval_suite="location-management",
        dataset_name="weather-agent-location-management-v1",
        git_sha="abc123",
    )

    assert metadata["application"] == "weather-agent-preprod-evals"
    assert metadata["langsmith_project"] == "weather-agent-preprod"
    assert metadata["environment"] == "staging"
    assert metadata["run_source"] == "local"
    assert metadata["ci"] is False


def test_langsmith_eval_experiment_prefix_is_filterable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LANGSMITH_APPLICATION", "weather-agent-preprod")

    prefix = build_langsmith_eval_experiment_prefix(
        eval_suite="notification-rule-proposal",
        model_provider="openrouter",
        model_name="qwen/qwen3.5-flash-02-23",
    )

    assert prefix == (
        "weather-agent-preprod--ci-eval--notification-rule-proposal--"
        "openrouter--qwen-qwen3.5-flash-02-23"
    )
