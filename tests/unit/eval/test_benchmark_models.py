from __future__ import annotations

import importlib.util
import sys
from datetime import UTC, datetime
from pathlib import Path

_BENCHMARK_PATH = Path(__file__).resolve().parents[3] / "scripts" / "eval" / "benchmark_models.py"
_CONFIG_PATH = (
    Path(__file__).resolve().parents[3] / "scripts" / "eval" / "model_benchmark_candidates.json"
)
_SPEC = importlib.util.spec_from_file_location("benchmark_models", _BENCHMARK_PATH)
assert _SPEC is not None
benchmark_models = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
sys.modules["benchmark_models"] = benchmark_models
_SPEC.loader.exec_module(benchmark_models)

BenchmarkConfig = benchmark_models.BenchmarkConfig
BenchmarkSummary = benchmark_models.BenchmarkSummary
DatasetSummary = benchmark_models.DatasetSummary
ExperimentSummary = benchmark_models.ExperimentSummary
LatencyGateSummary = benchmark_models.LatencyGateSummary
LatencyObservation = benchmark_models.LatencyObservation
expand_runs = benchmark_models.expand_runs
load_config = benchmark_models.load_config
render_summary = benchmark_models.render_summary
_latency_probe_cases = benchmark_models._latency_probe_cases
_percentile = benchmark_models._percentile


def test_load_config_reads_paid_and_free_candidates() -> None:
    config = load_config(_CONFIG_PATH)
    labels = {candidate.model_name for candidate in config.candidates}

    assert config.dataset_name == "weather-agent-weather-functional-v5"
    assert config.application_tag == "weather-agent-dev"
    assert config.latency_gate_enabled is True
    assert config.latency_gate_threshold_seconds == 10.0
    assert "ibm-granite/granite-4.1-8b" in labels
    assert "openai/gpt-oss-20b" in labels
    assert "openrouter/owl-alpha" in labels


def test_expand_runs_creates_price_and_latency_runs_for_paid_candidates() -> None:
    config = load_config(_CONFIG_PATH)

    runs = expand_runs(config, include_free=False, model_filter=frozenset({"openai/gpt-oss-20b"}))

    assert {run.routing_sort for run in runs} == {"price", "latency"}
    assert all(run.production_eligible for run in runs)


def test_expand_runs_excludes_free_candidates_by_default() -> None:
    config = load_config(_CONFIG_PATH)

    runs = expand_runs(config, include_free=False, model_filter=frozenset())

    assert "openrouter/owl-alpha" not in {run.model_name for run in runs}


def test_expand_runs_can_include_free_candidates() -> None:
    config = load_config(_CONFIG_PATH)

    runs = expand_runs(
        config,
        include_free=True,
        model_filter=frozenset({"openrouter/owl-alpha"}),
    )

    assert len(runs) == 1
    assert runs[0].production_eligible is False
    assert runs[0].routing_sort is None


def test_render_summary_links_saved_langsmith_experiments() -> None:
    summary = BenchmarkSummary(
        generated_at=datetime(2026, 5, 1, tzinfo=UTC),
        dataset=DatasetSummary(
            name="weather-agent-weather-functional-v2",
            id="dataset-id",
            url="https://smith.langchain.com/datasets/dataset-id",
            example_count=48,
            application_tag="weather-agent-dev",
        ),
        dataset_name="weather-agent-weather-functional-v2",
        repeats=3,
        max_concurrency=4,
        config_path="config.json",
        latency_gate=(
            LatencyGateSummary(
                label="openrouter:openai/gpt-4.1-nano:latency",
                included=True,
                threshold_seconds=10.0,
                p95_latency_seconds=1.2,
                max_latency_seconds=1.4,
                observations=(
                    LatencyObservation(
                        example_id="grounding-001",
                        latency_seconds=1.2,
                    ),
                ),
            ),
        ),
        experiments=(
            ExperimentSummary(
                label="openrouter:openai/gpt-4.1-nano:latency",
                provider="openrouter",
                model_name="openai/gpt-4.1-nano",
                production_eligible=True,
                routing_sort="latency",
                experiment_name="weather-functional-openrouter-openai/gpt-4.1-nano",
                experiment_id="experiment-id",
                url="https://smith.langchain.com/example",
                notes="Known cheap baseline.",
            ),
        ),
    )

    rendered = render_summary(summary, Path("reports/eval/model-benchmark.json"))

    assert "Saved summary: reports/eval/model-benchmark.json" in rendered
    assert "Dataset: weather-agent-weather-functional-v2 (dataset-id)" in rendered
    assert "Application tag: weather-agent-dev" in rendered
    assert "| openrouter:openai/gpt-4.1-nano:latency | yes | 1.20s | 1.40s |" in rendered
    assert "https://smith.langchain.com/example" in rendered
    assert "openrouter:openai/gpt-4.1-nano:latency" in rendered


def test_latency_probe_cases_are_representative_subset() -> None:
    cases = _latency_probe_cases(4)

    assert [case["id"] for case in cases] == [
        "grounding-001",
        "grounding-002",
        "grounding-007",
        "grounding-014",
    ]


def test_percentile_returns_none_for_empty_values() -> None:
    assert _percentile([], 0.95) is None


def test_percentile_uses_sorted_values() -> None:
    assert _percentile([5.0, 1.0, 3.0], 0.95) == 5.0
