from __future__ import annotations

import subprocess
import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import ModuleType

import pytest

SCRIPT_PATH = Path(__file__).resolve().parents[3] / "scripts/eval/run_model_quality_evals.py"


def _load_script() -> ModuleType:
    spec = spec_from_file_location("run_model_quality_evals", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_main_fails_with_clear_error_when_required_env_is_missing(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    run_model_quality_evals = _load_script()
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
    monkeypatch.delenv("WEATHER_AGENT_MODEL__API_KEY", raising=False)

    with pytest.raises(SystemExit) as exc_info:
        run_model_quality_evals.main()

    assert exc_info.value.code == 1
    assert (
        "missing required environment variables: LANGSMITH_API_KEY, WEATHER_AGENT_MODEL__API_KEY"
    ) in capsys.readouterr().err


def test_main_runs_all_eval_steps_in_order(monkeypatch: pytest.MonkeyPatch) -> None:
    run_model_quality_evals = _load_script()
    calls: list[tuple[list[str], Path, dict[str, str]]] = []

    monkeypatch.setenv("LANGSMITH_API_KEY", "ls-key")
    monkeypatch.setenv("WEATHER_AGENT_MODEL__API_KEY", "model-key")
    monkeypatch.delenv("WEATHER_AGENT_MODEL__PROVIDER", raising=False)
    monkeypatch.delenv("WEATHER_AGENT_MODEL__MODEL_NAME", raising=False)
    monkeypatch.delenv("WEATHER_AGENT_MODEL__BASE_URL", raising=False)

    def fake_run(
        command: list[str],
        *,
        cwd: Path,
        check: bool,
        env: dict[str, str],
    ) -> subprocess.CompletedProcess[str]:
        assert check is False
        calls.append((command, cwd, env))
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(run_model_quality_evals.subprocess, "run", fake_run)

    run_model_quality_evals.main()

    assert "WEATHER_AGENT_MODEL__PROVIDER" not in run_model_quality_evals.os.environ
    first_env = calls[0][2]
    assert first_env["WEATHER_AGENT_MODEL__PROVIDER"] == "openrouter"
    assert first_env["WEATHER_AGENT_MODEL__MODEL_NAME"] == "qwen/qwen3.5-flash-02-23"
    assert first_env["WEATHER_AGENT_MODEL__BASE_URL"] == "https://openrouter.ai/api/v1"
    assert calls == [
        ([sys.executable, script_path], run_model_quality_evals.REPO_ROOT, first_env)
        for _, script_path in run_model_quality_evals.EVAL_STEPS
    ]


def test_main_preserves_explicit_model_env(monkeypatch: pytest.MonkeyPatch) -> None:
    run_model_quality_evals = _load_script()

    monkeypatch.setenv("LANGSMITH_API_KEY", "ls-key")
    monkeypatch.setenv("WEATHER_AGENT_MODEL__API_KEY", "model-key")
    monkeypatch.setenv("WEATHER_AGENT_MODEL__PROVIDER", "openai")
    monkeypatch.setenv("WEATHER_AGENT_MODEL__MODEL_NAME", "gpt-4.1-mini")
    monkeypatch.setenv("WEATHER_AGENT_MODEL__BASE_URL", "https://proxy.example.com/v1")
    child_envs: list[dict[str, str]] = []

    def fake_run_step(_label: str, _script_path: str, env: dict[str, str]) -> None:
        child_envs.append(env)

    monkeypatch.setattr(run_model_quality_evals, "_run_step", fake_run_step)

    run_model_quality_evals.main()

    assert run_model_quality_evals.os.environ["WEATHER_AGENT_MODEL__PROVIDER"] == "openai"
    assert run_model_quality_evals.os.environ["WEATHER_AGENT_MODEL__MODEL_NAME"] == "gpt-4.1-mini"
    assert (
        run_model_quality_evals.os.environ["WEATHER_AGENT_MODEL__BASE_URL"]
        == "https://proxy.example.com/v1"
    )
    assert child_envs[0]["WEATHER_AGENT_MODEL__PROVIDER"] == "openai"
    assert child_envs[0]["WEATHER_AGENT_MODEL__MODEL_NAME"] == "gpt-4.1-mini"
    assert child_envs[0]["WEATHER_AGENT_MODEL__BASE_URL"] == "https://proxy.example.com/v1"


def test_main_stops_on_first_failed_eval_step(monkeypatch: pytest.MonkeyPatch) -> None:
    run_model_quality_evals = _load_script()
    calls: list[list[str]] = []

    monkeypatch.setenv("LANGSMITH_API_KEY", "ls-key")
    monkeypatch.setenv("WEATHER_AGENT_MODEL__API_KEY", "model-key")

    def fake_run(
        command: list[str],
        *,
        cwd: Path,
        check: bool,
        env: dict[str, str],
    ) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        return subprocess.CompletedProcess(command, 42)

    monkeypatch.setattr(run_model_quality_evals.subprocess, "run", fake_run)

    with pytest.raises(SystemExit) as exc_info:
        run_model_quality_evals.main()

    assert exc_info.value.code == 42
    assert calls == [[sys.executable, run_model_quality_evals.EVAL_STEPS[0][1]]]
