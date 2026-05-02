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
    calls: list[tuple[list[str], Path]] = []

    monkeypatch.setenv("LANGSMITH_API_KEY", "ls-key")
    monkeypatch.setenv("WEATHER_AGENT_MODEL__API_KEY", "model-key")

    def fake_run(
        command: list[str],
        *,
        cwd: Path,
        check: bool,
    ) -> subprocess.CompletedProcess[str]:
        assert check is False
        calls.append((command, cwd))
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(run_model_quality_evals.subprocess, "run", fake_run)

    run_model_quality_evals.main()

    assert calls == [
        ([sys.executable, script_path], run_model_quality_evals.REPO_ROOT)
        for _, script_path in run_model_quality_evals.EVAL_STEPS
    ]


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
    ) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        return subprocess.CompletedProcess(command, 42)

    monkeypatch.setattr(run_model_quality_evals.subprocess, "run", fake_run)

    with pytest.raises(SystemExit) as exc_info:
        run_model_quality_evals.main()

    assert exc_info.value.code == 42
    assert calls == [[sys.executable, run_model_quality_evals.EVAL_STEPS[0][1]]]
