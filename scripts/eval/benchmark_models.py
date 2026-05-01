"""Run configured model candidates as LangSmith experiments.

LangSmith is the system of record for benchmark results: each candidate becomes
an experiment with model/routing metadata, repetitions, traces, feedback scores,
latency, token, and cost columns. This script only orchestrates the sweep and
saves experiment URLs for quick return visits.

Usage:
    LANGSMITH_API_KEY=... WEATHER_AGENT_MODEL__API_KEY=... \
        uv run python scripts/eval/benchmark_models.py
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, Protocol, cast
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from langsmith import Client
from langsmith.run_helpers import tracing_context
from pydantic import BaseModel, ConfigDict, Field, SecretStr

from weather_agent.eval.dataset_gen import GeneratedCase, generate_cases
from weather_agent.eval.evaluators import weather_functional_correctness
from weather_agent.eval.targets import build_weather_answer_async_target_from_factory
from weather_agent.llm.model_factory import ModelFactory
from weather_agent.observability.langsmith_tracing import configure_tracing
from weather_agent.settings import LangSmithSettings, ModelSettings

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "scripts" / "eval" / "model_benchmark_candidates.json"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "reports" / "eval"
RoutingSort = Literal["price", "latency", "throughput"]
LangSmithResourceType = Literal["dataset", "experiment"]


class _DatasetLike(Protocol):
    id: object
    url: str | None


class ModelCandidateConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider: str
    model_name: str
    production_eligible: bool = True
    routing_sorts: tuple[RoutingSort | None, ...] = (None,)
    base_url: str | None = None
    notes: str = ""


class BenchmarkConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset_name: str = "weather-agent-weather-functional-v5"
    application_tag: str | None = "weather-agent-dev"
    experiment_prefix: str = "weather-functional"
    repeats: int = Field(default=3, ge=1)
    max_concurrency: int = Field(default=4, ge=0)
    latency_gate_enabled: bool = True
    latency_gate_example_count: int = Field(default=6, ge=1)
    latency_gate_threshold_seconds: float = Field(default=10.0, gt=0)
    full_eval_example_timeout_seconds: float = Field(default=30.0, gt=0)
    output_dir: Path = DEFAULT_OUTPUT_DIR
    candidates: tuple[ModelCandidateConfig, ...]


class BenchmarkRun(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider: str
    model_name: str
    production_eligible: bool
    routing_sort: RoutingSort | None
    base_url: str | None = None
    notes: str = ""

    @property
    def label(self) -> str:
        routing = self.routing_sort or "default"
        return f"{self.provider}:{self.model_name}:{routing}"


class ExperimentSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    label: str
    provider: str
    model_name: str
    production_eligible: bool
    routing_sort: RoutingSort | None
    experiment_name: str
    experiment_id: str
    url: str | None
    notes: str


class DatasetSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    id: str
    url: str | None
    example_count: int
    application_tag: str | None


class LatencyObservation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    example_id: str
    latency_seconds: float
    timed_out: bool = False
    error: str | None = None


class LatencyGateSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    label: str
    included: bool
    threshold_seconds: float
    p95_latency_seconds: float | None
    max_latency_seconds: float | None
    observations: tuple[LatencyObservation, ...]


class BenchmarkSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    generated_at: datetime
    dataset: DatasetSummary
    dataset_name: str
    repeats: int
    max_concurrency: int
    config_path: str
    latency_gate: tuple[LatencyGateSummary, ...]
    experiments: tuple[ExperimentSummary, ...]


def load_config(path: Path) -> BenchmarkConfig:
    return BenchmarkConfig.model_validate_json(path.read_text(encoding="utf-8"))


def expand_runs(
    config: BenchmarkConfig,
    *,
    include_free: bool,
    model_filter: frozenset[str],
) -> tuple[BenchmarkRun, ...]:
    runs: list[BenchmarkRun] = []
    for candidate in config.candidates:
        if not include_free and not candidate.production_eligible:
            continue
        if model_filter and candidate.model_name not in model_filter:
            continue
        for routing_sort in candidate.routing_sorts:
            runs.append(
                BenchmarkRun(
                    provider=candidate.provider,
                    model_name=candidate.model_name,
                    production_eligible=candidate.production_eligible,
                    routing_sort=routing_sort,
                    base_url=candidate.base_url,
                    notes=candidate.notes,
                )
            )
    return tuple(runs)


def _case_to_inputs(case: GeneratedCase) -> dict[str, object]:
    return {
        "id": case["id"],
        "question": case["question"],
        "current_time": case["current_time"],
        "frozen_facts": case["frozen_facts"],
        "hourly_values": case.get("hourly_values"),
        "target_hour": case.get("target_hour"),
        "expected_target_time": case.get("expected_target_time"),
    }


def _case_to_example(case: GeneratedCase) -> dict[str, object]:
    return {
        "inputs": {
            **_case_to_inputs(case),
        },
        "outputs": {
            "expected_facts": case["frozen_facts"],
            "required_location": True,
            "requested_attributes": case["requested_attributes"],
        },
    }


def _dataset_examples(cases: list[GeneratedCase]) -> list[dict[str, object]]:
    return [_case_to_example(case) for case in cases]


def _dataset_url(dataset: _DatasetLike) -> str | None:
    return dataset.url


def _dataset_id(dataset: _DatasetLike) -> str:
    return str(dataset.id)


def ensure_dataset(client: Client, dataset_name: str) -> DatasetSummary:
    cases = generate_cases()
    try:
        dataset = client.read_dataset(dataset_name=dataset_name)
    except Exception:
        dataset = client.create_dataset(
            dataset_name=dataset_name,
            description=(
                "Weather functional correctness benchmark. The real DeepAgent uses "
                "fixture-backed tools and must answer with required weather facts. "
                "Covers current conditions and forecast periods with explicit hours."
            ),
        )
        client.create_examples(dataset_id=dataset.id, examples=_dataset_examples(cases))
    return DatasetSummary(
        name=dataset_name,
        id=_dataset_id(dataset),
        url=_dataset_url(dataset),
        example_count=len(cases),
        application_tag=None,
    )


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, round((len(ordered) - 1) * percentile)))
    return ordered[index]


def _latency_probe_cases(count: int) -> tuple[GeneratedCase, ...]:
    cases = generate_cases()
    preferred_ids = (
        "grounding-001",
        "grounding-002",
        "grounding-007",
        "grounding-014",
        "grounding-022",
        "grounding-030",
        "grounding-038",
        "grounding-046",
    )
    by_id = {case["id"]: case for case in cases}
    selected = [by_id[case_id] for case_id in preferred_ids if case_id in by_id]
    if len(selected) < count:
        selected.extend(case for case in cases if case not in selected)
    return tuple(selected[:count])


async def run_latency_gate(
    run: BenchmarkRun,
    *,
    api_key: str,
    example_count: int,
    threshold_seconds: float,
) -> LatencyGateSummary:
    settings = ModelSettings(
        provider=run.provider,
        model_name=run.model_name,
        temperature=0.0,
        api_key=SecretStr(api_key),
        base_url=run.base_url,
        routing_sort=run.routing_sort,
        require_supported_parameters=True,
    )
    target = build_weather_answer_async_target_from_factory(
        lambda: ModelFactory(settings).create_chat_model()
    )
    observations: list[LatencyObservation] = []
    for case in _latency_probe_cases(example_count):
        inputs = _case_to_inputs(case)
        started = time.perf_counter()
        try:
            with tracing_context(enabled=False):
                await asyncio.wait_for(target(inputs), timeout=threshold_seconds)
            latency_seconds = time.perf_counter() - started
            observations.append(
                LatencyObservation(
                    example_id=case["id"],
                    latency_seconds=latency_seconds,
                )
            )
        except TimeoutError:
            observations.append(
                LatencyObservation(
                    example_id=case["id"],
                    latency_seconds=threshold_seconds,
                    timed_out=True,
                    error=f"Timed out after {threshold_seconds:.1f}s",
                )
            )
            break
        except Exception as exc:
            observations.append(
                LatencyObservation(
                    example_id=case["id"],
                    latency_seconds=time.perf_counter() - started,
                    error=f"{type(exc).__name__}: {exc}",
                )
            )
            break

    latencies = [observation.latency_seconds for observation in observations]
    p95_latency = _percentile(latencies, 0.95)
    max_latency = max(latencies) if latencies else None
    included = (
        len(observations) == example_count
        and all(observation.error is None for observation in observations)
        and p95_latency is not None
        and p95_latency <= threshold_seconds
    )
    return LatencyGateSummary(
        label=run.label,
        included=included,
        threshold_seconds=threshold_seconds,
        p95_latency_seconds=p95_latency,
        max_latency_seconds=max_latency,
        observations=tuple(observations),
    )


async def run_langsmith_experiment(
    run: BenchmarkRun,
    *,
    config: BenchmarkConfig,
    api_key: str,
    langsmith_project: str,
    client: Client,
) -> ExperimentSummary:
    settings = ModelSettings(
        provider=run.provider,
        model_name=run.model_name,
        temperature=0.0,
        api_key=SecretStr(api_key),
        base_url=run.base_url,
        routing_sort=run.routing_sort,
        require_supported_parameters=True,
    )
    target = build_weather_answer_async_target_from_factory(
        lambda: ModelFactory(settings).create_chat_model()
    )

    async def timed_target(inputs: dict[str, object]) -> dict[str, object]:
        return await asyncio.wait_for(
            target(inputs),
            timeout=config.full_eval_example_timeout_seconds,
        )

    results = await client.aevaluate(
        timed_target,
        data=config.dataset_name,
        evaluators=[weather_functional_correctness],
        experiment_prefix=f"{config.experiment_prefix}-{run.provider}-{run.model_name}",
        description=run.notes,
        metadata={
            "benchmark": "provider_agnostic_model_selection",
            "model_provider": run.provider,
            "model_name": run.model_name,
            "models": [run.model_name],
            "routing_sort": run.routing_sort or "default",
            "production_eligible": run.production_eligible,
            "langsmith_project": langsmith_project,
            "full_eval_example_timeout_seconds": config.full_eval_example_timeout_seconds,
        },
        max_concurrency=config.max_concurrency,
        num_repetitions=config.repeats,
        error_handling="log",
    )
    async for _ in results:
        pass
    summary = ExperimentSummary(
        label=run.label,
        provider=run.provider,
        model_name=run.model_name,
        production_eligible=run.production_eligible,
        routing_sort=run.routing_sort,
        experiment_name=results.experiment_name,
        experiment_id=str(results.experiment_id),
        url=await results.get_comparison_url(),
        notes=run.notes,
    )
    return summary


class LangSmithResourceTagger:
    def __init__(self, *, api_key: str, api_url: str = "https://api.smith.langchain.com") -> None:
        self._api_key = api_key
        self._api_url = api_url.rstrip("/")

    def tag_application(
        self,
        resource_type: LangSmithResourceType,
        resource_id: str,
        application: str,
    ) -> None:
        tag_value_id = self._application_tag_value_id(application)
        if self._has_tag(resource_type, resource_id, tag_value_id):
            return
        self._post_json(
            "/api/v1/workspaces/current/taggings",
            {
                "tag_value_id": tag_value_id,
                "resource_type": resource_type,
                "resource_id": resource_id,
            },
        )

    def _application_tag_value_id(self, application: str) -> str:
        tags = self._get_json("/api/v1/workspaces/current/tags")
        if not isinstance(tags, list):
            raise RuntimeError("LangSmith tags API returned an unexpected payload.")
        for tag in tags:
            if not isinstance(tag, dict) or tag.get("key") != "Application":
                continue
            for value in tag.get("values", []):
                if isinstance(value, dict) and value.get("value") == application:
                    tag_value_id = value.get("id")
                    if isinstance(tag_value_id, str):
                        return tag_value_id
        raise RuntimeError(
            f"LangSmith Application tag '{application}' does not exist. "
            "Create it in Settings -> Resource tags, or set application_tag to null."
        )

    def _has_tag(
        self,
        resource_type: LangSmithResourceType,
        resource_id: str,
        tag_value_id: str,
    ) -> bool:
        path = (
            "/api/v1/workspaces/current/tags/resource"
            f"?resource_type={resource_type}&resource_id={resource_id}"
        )
        tags = self._get_json(path)
        if not isinstance(tags, list):
            return False
        for tag in tags:
            if not isinstance(tag, dict):
                continue
            for value in tag.get("values", []):
                if isinstance(value, dict) and value.get("id") == tag_value_id:
                    return True
        return False

    def _get_json(self, path: str) -> object:
        return self._request_json("GET", path)

    def _post_json(self, path: str, payload: dict[str, str]) -> object:
        return self._request_json("POST", path, payload)

    def _request_json(
        self,
        method: Literal["GET", "POST"],
        path: str,
        payload: dict[str, str] | None = None,
    ) -> object:
        body = None if payload is None else json.dumps(payload).encode("utf-8")
        request = Request(
            f"{self._api_url}{path}",
            data=body,
            headers={
                "X-API-Key": self._api_key,
                "Content-Type": "application/json",
            },
            method=method,
        )
        try:
            with urlopen(request, timeout=20) as response:
                return cast(object, json.loads(response.read().decode("utf-8")))
        except HTTPError as exc:
            details = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"LangSmith resource tagging failed ({exc.code}): {details}"
            ) from exc


def save_summary(
    summary: BenchmarkSummary,
    *,
    output_dir: Path,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = summary.generated_at.strftime("%Y%m%dT%H%M%SZ")
    output_path = output_dir / f"model-benchmark-{timestamp}.json"
    output_path.write_text(summary.model_dump_json(indent=2), encoding="utf-8")
    return output_path


def render_summary(summary: BenchmarkSummary, output_path: Path) -> str:
    lines = [
        f"Saved summary: {output_path}",
        f"Dataset: {summary.dataset.name} ({summary.dataset.id})",
        f"Dataset URL: {summary.dataset.url or 'unavailable'}",
        f"Application tag: {summary.dataset.application_tag or 'none'}",
        "",
        "| Latency gate | Included | P95 | Max |",
        "|---|---:|---:|---:|",
    ]
    for gate in summary.latency_gate:
        p95 = "n/a" if gate.p95_latency_seconds is None else f"{gate.p95_latency_seconds:.2f}s"
        max_latency = (
            "n/a" if gate.max_latency_seconds is None else f"{gate.max_latency_seconds:.2f}s"
        )
        lines.append(
            "| {label} | {included} | {p95} | {max_latency} |".format(
                label=gate.label,
                included="yes" if gate.included else "no",
                p95=p95,
                max_latency=max_latency,
            )
        )
    lines.extend(
        [
            "",
            "| Candidate | Prod | Routing | LangSmith experiment |",
            "|---|---:|---|---|",
        ]
    )
    for experiment in summary.experiments:
        url = experiment.url or experiment.experiment_id
        lines.append(
            "| {label} | {prod} | {routing} | {url} |".format(
                label=experiment.label,
                prod="yes" if experiment.production_eligible else "no",
                routing=experiment.routing_sort or "default",
                url=url,
            )
        )
    return "\n".join(lines)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--models", nargs="*", help="Optional model IDs to benchmark.")
    parser.add_argument("--include-free", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--skip-latency-gate", action="store_true")
    parser.add_argument(
        "--latency-gate-only",
        action="store_true",
        help=(
            "Run the local latency gate and save a summary without creating LangSmith experiments."
        ),
    )
    parser.add_argument("--latency-threshold-seconds", type=float)
    parser.add_argument("--full-eval-timeout-seconds", type=float)
    return parser.parse_args()


async def _run() -> None:
    args = _parse_args()
    langsmith_api_key = os.environ.get("LANGSMITH_API_KEY")
    model_api_key = os.environ.get("WEATHER_AGENT_MODEL__API_KEY")
    if not langsmith_api_key:
        print("Error: LANGSMITH_API_KEY is required", file=sys.stderr)
        sys.exit(1)
    if not model_api_key:
        print("Error: WEATHER_AGENT_MODEL__API_KEY is required", file=sys.stderr)
        sys.exit(1)

    langsmith_project = os.environ.get("LANGSMITH_PROJECT", "weather-agent-dev")
    configure_tracing(
        LangSmithSettings(
            enabled=True,
            api_key=SecretStr(langsmith_api_key),
            project=langsmith_project,
        )
    )

    config = load_config(args.config)
    if args.latency_threshold_seconds is not None:
        config = config.model_copy(
            update={"latency_gate_threshold_seconds": args.latency_threshold_seconds}
        )
    if args.full_eval_timeout_seconds is not None:
        config = config.model_copy(
            update={"full_eval_example_timeout_seconds": args.full_eval_timeout_seconds}
        )
    runs = expand_runs(
        config,
        include_free=args.include_free,
        model_filter=frozenset(args.models or ()),
    )
    client = Client()

    dataset = ensure_dataset(client, config.dataset_name)
    tagger = (
        LangSmithResourceTagger(
            api_key=langsmith_api_key,
            api_url=os.environ.get("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com"),
        )
        if config.application_tag is not None
        else None
    )
    application_tag = config.application_tag
    if tagger is not None and application_tag is not None:
        tagger.tag_application("dataset", dataset.id, application_tag)
        dataset = dataset.model_copy(update={"application_tag": application_tag})

    latency_gate: tuple[LatencyGateSummary, ...] = ()
    selected_runs = runs
    if config.latency_gate_enabled and not args.skip_latency_gate:
        gate_results = [
            await run_latency_gate(
                run,
                api_key=model_api_key,
                example_count=config.latency_gate_example_count,
                threshold_seconds=config.latency_gate_threshold_seconds,
            )
            for run in runs
        ]
        latency_gate = tuple(gate_results)
        included_labels = {result.label for result in gate_results if result.included}
        selected_runs = tuple(run for run in runs if run.label in included_labels)

    if args.latency_gate_only:
        summary = BenchmarkSummary(
            generated_at=datetime.now(UTC),
            dataset=dataset,
            dataset_name=config.dataset_name,
            repeats=config.repeats,
            max_concurrency=config.max_concurrency,
            config_path=str(args.config),
            latency_gate=latency_gate,
            experiments=(),
        )
        output_path = save_summary(summary, output_dir=args.output_dir or config.output_dir)
        print(render_summary(summary, output_path))
        return

    experiments: list[ExperimentSummary] = []
    for run in selected_runs:
        experiment = await run_langsmith_experiment(
            run,
            config=config,
            api_key=model_api_key,
            langsmith_project=langsmith_project,
            client=client,
        )
        if tagger is not None and application_tag is not None:
            tagger.tag_application("experiment", experiment.experiment_id, application_tag)
        experiments.append(experiment)

    summary = BenchmarkSummary(
        generated_at=datetime.now(UTC),
        dataset=dataset,
        dataset_name=config.dataset_name,
        repeats=config.repeats,
        max_concurrency=config.max_concurrency,
        config_path=str(args.config),
        latency_gate=latency_gate,
        experiments=tuple(experiments),
    )
    output_path = save_summary(summary, output_dir=args.output_dir or config.output_dir)
    print(render_summary(summary, output_path))


def main() -> None:
    asyncio.run(_run())


if __name__ == "__main__":
    main()
