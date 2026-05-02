"""Architecture and determinism guardrails for the runtime refactor.

These tests codify project-wide invariants so they are enforced by ``pytest``
and cannot regress silently:

1.  No production Python file exceeds 300 lines (with documented exceptions).
2.  Domain-layer modules never import infrastructure, adapters, llm, or
    deepagents — they must remain pure business logic.
3.  Application-layer modules never import adapters/telegram directly.
4.  Runtime rule expression/rule evaluation paths never call the LLM or model agent.
"""

from __future__ import annotations

from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[3] / "src" / "weather_agent"

# Files that are deliberately longer than 300 lines, with a reason for each.
_LINE_COUNT_EXCEPTIONS: dict[str, str] = {
    "domain/rule_expression/evaluator.py": (
        "Complex rule expression evaluation with multiple visitor types"
    ),
    "domain/rule_expression/validation.py": "rule expression syntax validation and error recovery",
    "domain/notifications/events.py": "Notification event lifecycle management",
    "domain/locations.py": "Location CRUD with search, alias matching, defaults",
    "infrastructure/worker/rule_evaluator.py": "Rule evaluation worker with scheduling",
    "infrastructure/db/base.py": "SQLAlchemy ORM model declarations",
    "adapters/telegram/bot.py": "Telegram bot setup with command registration",
    "adapters/imgw/warnings_provider.py": "IMGW warning parsing and transformation",
    "adapters/open_meteo/forecast_provider.py": "Open-Meteo API forecast fetching",
    "eval/targets.py": "Weather grounding eval fixture target with sync and async entrypoints",
    "eval/location_management_targets.py": "Location management eval target orchestration",
    "llm/tools/weather_tools.py": "Weather toolbox with forecast/observations/location",
    "llm/tools/rules_tools.py": (
        "Rules toolbox with notification rule CRUD and rule expression validation"
    ),
}

# ---------------------------------------------------------------------------
# 1. File-size guardrail
# ---------------------------------------------------------------------------


def _all_py_files() -> list[Path]:
    return sorted(SRC.rglob("*.py"))


@pytest.mark.parametrize("pyfile", _all_py_files(), ids=lambda p: str(p.relative_to(SRC)))
def test_production_file_stays_under_300_lines(pyfile: Path) -> None:
    rel = str(pyfile.relative_to(SRC))
    if rel in _LINE_COUNT_EXCEPTIONS:
        pytest.skip(f"Exempt: {_LINE_COUNT_EXCEPTIONS[rel]}")
    lines = pyfile.read_text().splitlines()
    assert len(lines) <= 300, f"{rel} has {len(lines)} lines (limit 300)"


# ---------------------------------------------------------------------------
# 2. Domain-layer import boundaries
# ---------------------------------------------------------------------------

_FORBIDDEN_DOMAIN_IMPORTS: dict[str, list[str]] = {
    "domain": [
        "weather_agent.infrastructure",
        "weather_agent.adapters",
        "weather_agent.llm",
        "deepagents",
    ],
}


def _imports_from(text: str, prefix: str) -> list[str]:
    """Return import lines in *text* that reference *prefix*."""
    result: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("import ") or stripped.startswith("from "):
            if prefix in stripped:
                result.append(stripped)
    return result


# Domain modules that are known to import infrastructure (ORM models).
# These are pre-existing — the ideal fix is to move ORM access behind
# repository/service boundaries, but that is out of scope for this
# architecture guardrail layer.
_DOMAIN_BOUNDARY_EXCEPTIONS: frozenset[str] = frozenset(
    {
        "domain/locations.py",
        "domain/notifications/deduplication.py",
        "domain/notifications/events.py",
        "domain/rules/service.py",
    }
)


@pytest.mark.parametrize("pyfile", _all_py_files(), ids=lambda p: str(p.relative_to(SRC)))
def test_domain_import_boundaries(pyfile: Path) -> None:
    rel = str(pyfile.relative_to(SRC))
    if not rel.startswith("domain/"):
        pytest.skip("Not a domain module")
    if rel in _DOMAIN_BOUNDARY_EXCEPTIONS:
        pytest.skip(f"Known exception: {rel}")
    text = pyfile.read_text()
    for prefix in _FORBIDDEN_DOMAIN_IMPORTS["domain"]:
        matches = _imports_from(text, prefix)
        assert not matches, f"{rel} must not import '{prefix}':\n  " + "\n  ".join(matches)


# ---------------------------------------------------------------------------
# 3. Application-layer adapters boundary
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("pyfile", _all_py_files(), ids=lambda p: str(p.relative_to(SRC)))
def test_application_does_not_import_telegram(pyfile: Path) -> None:
    rel = str(pyfile.relative_to(SRC))
    if not rel.startswith("application/"):
        pytest.skip("Not an application module")
    text = pyfile.read_text()
    matches = _imports_from(text, "weather_agent.adapters.telegram")
    assert not matches, f"{rel} must not import telegram adapter directly:\n  " + "\n  ".join(
        matches
    )


# ---------------------------------------------------------------------------
# 4. Deterministic rule expression/rule runtime — never calls LLM or model agent
# ---------------------------------------------------------------------------


def _all_py_files_under(*parts: str) -> list[Path]:
    return sorted((SRC / parts[0]).rglob("*.py"))


@pytest.mark.parametrize(
    "pyfile",
    [p for p in _all_py_files() if "rule" in str(p).lower() or "rule_expression" in str(p).lower()],
    ids=lambda p: str(p.relative_to(SRC)),
)
def test_rule_runtime_does_not_call_llm(pyfile: Path) -> None:
    """Runtime rule/evaluation files must not import llm or deepagents.

    LLM may *propose* rule expression expressions, but the validation, persistence, and
    evaluation paths must remain purely deterministic.
    """
    rel = str(pyfile.relative_to(SRC))
    text = pyfile.read_text()
    for forbidden in ("weather_agent.llm", "weather_agent.agent_factory", "deepagents"):
        matches = _imports_from(text, forbidden)
        if matches:
            # Allow llm imports in rules/prompt construction modules that are
            # clearly used only at proposal time, not runtime.
            if "proposal" in rel.lower():
                continue
            pytest.fail(f"{rel} imports {forbidden}")


# ---------------------------------------------------------------------------
# 5. Infrastructure not importing adapters
# ---------------------------------------------------------------------------


_INFRASTRUCTURE_BOUNDARY_EXCEPTIONS: frozenset[str] = frozenset(
    {
        "infrastructure/app_container.py",
        "infrastructure/memory/thread_memory.py",
    }
)


@pytest.mark.parametrize("pyfile", _all_py_files(), ids=lambda p: str(p.relative_to(SRC)))
def test_infrastructure_does_not_import_adapters(pyfile: Path) -> None:
    rel = str(pyfile.relative_to(SRC))
    if not rel.startswith("infrastructure/"):
        pytest.skip("Not an infrastructure module")
    if rel in _INFRASTRUCTURE_BOUNDARY_EXCEPTIONS:
        pytest.skip(f"Known exception: {rel}")
    text = pyfile.read_text()
    matches = _imports_from(text, "weather_agent.adapters")
    if matches:
        pytest.fail(f"{rel} imports adapter:\n  " + "\n  ".join(matches))
