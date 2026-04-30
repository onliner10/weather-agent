# LangSmith Evaluation Reference Implementation

## Metadata
```yaml
id: langsmith-evaluation-reference-implementation
owner: Mateusz Urban
status: draft
created: 2026-04-30
updated: 2026-04-30
scope: Weather Agent LLM evaluation, CI gates, and production quality monitoring
```

## Goal

Build a production-grade evaluation system for the Telegram weather bot using LangSmith as the experiment, tracing, and feedback platform.

The system should prove that the bot:

- Answers weather questions using grounded weather data.
- Extracts intent, location, time range, and rule requests correctly.
- Refuses or redirects non-weather questions.
- Handles ambiguous user input safely.
- Handles tool, model, and provider failures gracefully.
- Improves over time by turning real production failures into regression tests.

This is a learning project, but the implementation should look like something expected from a Senior AI Application Engineer working on a serious production system.

## Evaluation Philosophy

Use different evaluators for different risks.

| Risk | Best evaluator | Gate type |
| --- | --- | --- |
| Wrong intent | Deterministic label check | CI hard gate |
| Wrong location | Deterministic label check | CI hard gate |
| Wrong time range | Deterministic label check | CI hard gate |
| Wrong weather fact | Programmatic check against frozen weather snapshot | CI hard gate |
| Unsupported weather claim | Programmatic check plus AI judge | CI hard gate |
| Off-topic answer | Deterministic category check plus AI judge | CI hard gate |
| Poor Polish style | AI judge | CI soft gate at first |
| Low helpfulness | AI judge | CI soft gate or production metric |
| Latency regression | Timed run metric | CI soft gate and production alert |
| Cost regression | Token/cost metric | CI soft gate and production metric |
| Provider instability | Metrics and traces | Production alert |

Do not use one aggregate score as the release decision. Report scores by slice: weather QA, rule creation, ambiguity, provider failure, off-topic, and command handling.

## Step 1: Define The Quality Contract

Write the product-level quality contract before implementing eval code.

Required behavior:

- Weather questions should result in a grounded answer or a clarification request.
- Ambiguous location should produce a Polish clarification question instead of guessing.
- Ambiguous time range should use a documented default or ask a clarification question.
- Forecast answers should not invent values absent from tool output.
- Off-topic questions should be refused briefly and redirected to weather capabilities.
- Rule requests should produce valid, allowlisted CEL or ask for missing details.
- Provider failures should produce a Polish user-facing apology with no fake forecast.
- The bot should never expose raw traces, secrets, prompts, internal CEL validation details, or stack traces.

Reference release rule:

- Safety and domain adherence must pass 100% on critical cases.
- Weather grounding must pass all critical cases.
- Non-critical helpfulness can be tracked as a trend before it becomes a hard gate.

## Step 2: Create LangSmith Datasets

Create separate LangSmith datasets instead of one mixed benchmark. This keeps failure diagnosis clear.

Recommended dataset names:

| Dataset | Purpose |
| --- | --- |
| `weather-agent-intent-routing-v1` | Intent classification and flow routing |
| `weather-agent-weather-grounding-v1` | Weather answers against frozen tool outputs |
| `weather-agent-location-time-v1` | Location and time-range extraction |
| `weather-agent-domain-boundary-v1` | Off-topic, mixed-topic, and adversarial prompts |
| `weather-agent-rule-cel-v1` | Rule creation and CEL generation |
| `weather-agent-failure-recovery-v1` | Provider/model/tool failures |
| `weather-agent-production-regressions-v1` | Real failures promoted from production traces |

Start by migrating the existing cases from `tests/eval/dataset.py` into these datasets. Keep the local Python dataset as a fast offline source of truth, but make LangSmith the place where experiments, comparisons, and review history are visible.

## Step 3: Use A Typed Example Schema

Each example should be explicit about what is being evaluated.

Recommended input schema:

```json
{
  "message": "jaka będzie pogoda w weekend nad Jeziorakiem?",
  "conversation_context": {
    "known_locations": ["Dom", "Chwarzno", "Jeziorak"],
    "default_location": null,
    "timezone": "Europe/Warsaw",
    "current_time": "2026-04-30T12:00:00+02:00"
  },
  "frozen_tool_outputs": {
    "forecast": {
      "location": "Jeziorak",
      "period": "weekend",
      "summary_facts": {
        "rain_expected": true,
        "max_wind_gusts_ms": 11.5,
        "temperature_min_c": 7.0,
        "temperature_max_c": 15.0
      }
    }
  }
}
```

Recommended reference output schema:

```json
{
  "expected_intent": "weather",
  "expected_location": "Jeziorak",
  "expected_time_range": "weekend",
  "expected_tool_calls": ["get_forecast"],
  "expected_facts": {
    "rain_expected": true,
    "wind_severity": "moderate",
    "temperature_band": "cool"
  },
  "must_clarify": false,
  "must_refuse": false,
  "allowed_response_language": "pl"
}
```

For off-topic examples, set `must_refuse=true` and include the expected redirect behavior.

For provider failure examples, include the injected failure and assert that the response does not invent weather.

## Step 4: Build The Offline Evaluation Harness

Create a single evaluation entrypoint, for example:

```bash
uv run python scripts/eval/langsmith_eval.py --suite smoke
uv run python scripts/eval/langsmith_eval.py --suite ci
uv run python scripts/eval/langsmith_eval.py --suite full
```

The harness should:

- Load examples from LangSmith or a local fixture export.
- Run the bot through a deterministic test adapter, not real Telegram.
- Inject frozen weather provider responses.
- Disable live provider calls unless explicitly running smoke tests.
- Return structured outputs: intent, location, time range, tool calls, response text, refusal flag, clarification flag, and extracted answer facts.
- Log every run as a LangSmith experiment with model name, prompt version, git SHA, and dataset version metadata.

Do not evaluate only the final natural-language response. Capture intermediate structured decisions because they are easier to score and debug.

## Step 5: Implement Deterministic Evaluators

Implement deterministic evaluators first. They are the backbone of the quality system.

Required deterministic evaluators:

| Evaluator | What it checks | Output key |
| --- | --- | --- |
| Intent correctness | Predicted intent equals reference intent | `intent_correct` |
| Location correctness | Resolved location equals reference location | `location_correct` |
| Time correctness | Resolved time range equals reference range | `time_range_correct` |
| Tool correctness | Tool names and arguments match expectation | `tool_call_correct` |
| Weather fact correctness | Extracted answer facts match frozen snapshot | `weather_fact_correct` |
| No unsupported claims | Response does not mention facts absent from snapshot | `no_unsupported_claims` |
| Refusal correctness | Off-topic examples are refused and weather examples are not | `refusal_correct` |
| CEL validity | Generated CEL parses and uses allowlisted metrics/functions | `cel_valid` |
| Provider failure behavior | Failure examples avoid fake forecasts and apologize in Polish | `failure_recovery_correct` |

These evaluators should return booleans or numeric scores and short explanations. Explanations matter because LangSmith should help diagnose failures quickly.

## Step 6: Add AI-Judge Evaluators Carefully

Use AI judges for semantic qualities that are hard to check with exact rules.

Recommended AI-judge evaluators:

| Evaluator | What it judges | Gate |
| --- | --- | --- |
| Groundedness judge | Whether answer is supported by frozen tool facts | CI hard gate after calibration |
| Relevance judge | Whether answer addresses the user request | CI soft gate, later hard for critical cases |
| Helpfulness judge | Whether answer is useful and actionable | Production metric, optional CI soft gate |
| Polish style judge | Whether answer is natural Polish and concise | Production metric, optional CI soft gate |
| Domain-boundary judge | Whether off-topic response refused and redirected correctly | CI hard gate after calibration |
| Clarification judge | Whether clarification question asks for the missing information | CI hard gate for ambiguity cases |

Judge prompt requirements:

- Force structured JSON output.
- Include the user message, frozen facts, reference expectations, and model response.
- Ask for a score and a short reason.
- Tell the judge not to reward unsupported detail.
- Use temperature 0.
- Version every judge prompt.

Important rule: an AI judge must not be the only evaluator for weather fact correctness. Factual weather checks should be programmatic whenever possible.

## Step 7: Calibrate AI Judges Before Trusting Them

Before using an AI judge as a release gate:

1. Create 30 to 50 labeled examples for that judge.
2. Include obvious passes, obvious failures, and borderline cases.
3. Compare judge decisions to human labels.
4. Inspect false positives and false negatives.
5. Tighten the prompt and rubric.
6. Record the judge prompt version in LangSmith metadata.
7. Only promote it from soft metric to hard gate after it is stable.

Target calibration expectations:

- Critical safety/domain judge: near-zero false passes.
- Groundedness judge: low false passes, even if it is slightly strict.
- Helpfulness judge: acceptable as trend metric even if imperfect.

## Step 8: Define CI Quality Gates

CI should be stable, deterministic, and strict on critical behavior.

Recommended CI levels:

| Level | Command | When it runs | Purpose |
| --- | --- | --- | --- |
| Smoke | `--suite smoke` | every PR | Fast critical examples |
| CI | `--suite ci` | every PR touching prompts, tools, LLM flow, or rules | Main blocking gate |
| Full | `--suite full` | nightly or before release | Larger benchmark and pairwise comparisons |

Recommended initial thresholds:

| Metric | Smoke | CI | Full |
| --- | ---: | ---: | ---: |
| Critical off-topic refusal | 100% | 100% | 100% |
| Weather fact correctness | 100% | >= 98% | >= 98% |
| Unsupported-claim prevention | 100% | >= 99% | >= 99% |
| Intent correctness | 100% | >= 98% | >= 98% |
| Location correctness | 100% | >= 95% | >= 95% |
| Time-range correctness | 100% | >= 95% | >= 95% |
| CEL validity | 100% | >= 98% | >= 98% |
| Provider failure recovery | 100% | 100% | 100% |
| AI-judge groundedness | >= 95% | >= 95% | >= 95% |
| p95 turn latency on frozen eval | report only | no > 20% regression | no > 20% regression |
| estimated cost per turn | report only | no > 20% regression | no > 20% regression |

Keep the first thresholds intentionally strict for safety and grounding. If the model cannot meet them, improve the product or narrow the supported behavior instead of weakening the gate silently.

## Step 9: Add Pairwise Regression Tests

For prompt or model changes, run candidate output against the current baseline.

Use pairwise comparison for:

- answer helpfulness
- Polish fluency
- conciseness
- user preference
- ability to explain weather tradeoffs

Do not use pairwise comparison alone for release approval. A prettier answer that is less factual should lose because deterministic factual gates still block it.

Pairwise metadata should include:

- baseline model
- candidate model
- prompt version
- git SHA
- dataset version
- judge prompt version

## Step 10: Production Monitoring In LangSmith

Use LangSmith traces for qualitative debugging and sampled online evaluation.

Track these production metrics in LangSmith:

- trace volume by intent
- failure traces by node or tool
- model latency
- tool latency
- token usage and estimated cost
- sampled groundedness score
- sampled relevance score
- sampled domain-boundary score
- percentage of clarification responses
- percentage of refusal responses
- percentage of empty or failed responses

Use metadata fields that support slicing:

- `environment`: `dev`, `staging`, `production`
- `service`: `bot` or `worker`
- `intent`
- `model_provider`
- `model_name`
- `prompt_version`
- `flow_version`
- `tool_names`
- `has_provider_failure`
- `requires_clarification`
- `refusal_expected`
- `correlation_id`

Respect existing privacy constraints from `README.md`: do not send full raw message history, secrets, or raw provider payloads as metadata.

## Step 11: Production Metrics Outside LangSmith

Keep operational metrics in Prometheus/Grafana because they are better for alerting.

Already available metrics in `src/weather_agent/observability/metrics.py` should be dashboarded and alerted on:

- `weather_agent_conversation_turns_total`
- `weather_agent_conversation_failures_total`
- `weather_agent_conversation_turn_duration_seconds`
- `weather_agent_tool_calls_total`
- `weather_agent_tool_call_duration_seconds`
- `weather_agent_llm_requests_total`
- `weather_agent_llm_request_duration_seconds`
- `weather_agent_provider_requests_total`
- `weather_agent_provider_request_duration_seconds`
- `weather_agent_rule_evaluation_failures_total`
- `weather_agent_forecast_refresh_total`
- `weather_agent_last_successful_worker_cycle_timestamp_seconds`
- `weather_agent_last_successful_forecast_refresh_timestamp_seconds`

Recommended alert examples:

- Conversation failure rate above 5% for 10 minutes.
- Provider failure rate above 10% for 10 minutes.
- p95 conversation latency above 15 seconds for 10 minutes.
- No successful forecast refresh for more than 2 expected refresh intervals.
- No successful worker cycle for more than 2 expected cycle intervals.

## Step 12: Human Review Loop

Create a weekly review workflow:

1. Sample production traces from LangSmith.
2. Prioritize traces with failures, low judge scores, long latency, or user correction messages.
3. Manually label the root cause.
4. Add high-value failures to `weather-agent-production-regressions-v1`.
5. Add or update deterministic evaluators if the failure can be checked without an AI judge.
6. Re-run the baseline and candidate experiments.
7. Track whether the regression suite is growing and whether repeated failure modes disappear.

Suggested failure labels:

- `wrong_intent`
- `wrong_location`
- `wrong_time_range`
- `wrong_tool_call`
- `unsupported_claim`
- `bad_refusal`
- `missed_clarification`
- `provider_failure_bad_response`
- `cel_invalid`
- `polish_style_issue`
- `latency_issue`

This loop is what makes the project look production-grade: production failures become permanent tests.

## Step 13: Rollout Strategy

Use staged rollout for any prompt, model, or routing change.

Recommended flow:

1. Run local unit tests: `uv run pytest`.
2. Run static checks: `uv run ruff check .`, `uv run ruff format --check .`, `uv run mypy`.
3. Run eval smoke suite.
4. Run eval CI suite.
5. Run pairwise comparison against the current baseline for changed prompts/models.
6. Deploy to staging with LangSmith project `weather-agent-staging`.
7. Run a short manual Telegram script against staging.
8. Deploy to production with LangSmith project `weather-agent-prod`.
9. Watch production metrics and sampled online judge scores for the first hour.

Rollback rule:

- If critical factuality, domain-boundary, or provider-failure metrics regress, roll back the prompt/model/config change first.
- If operational metrics regress without eval regressions, inspect traces and logs by `correlation_id`.

## Step 14: Interview-Ready Evidence

Maintain artifacts that demonstrate engineering maturity:

- LangSmith datasets with clear names and versions.
- LangSmith experiment links for baseline and candidate runs.
- A CI job that fails on critical eval regressions.
- A table of thresholds and why each threshold exists.
- A dashboard with live latency, failure, and cost trends.
- A regression dataset grown from production traces.
- A judge calibration note showing that AI judges were validated before becoming hard gates.
- Example postmortems where a production failure became a new eval case.

In an interview, the strongest story is not "I used LangSmith". The strongest story is:

"I identified the bot's highest-risk failure modes, built deterministic checks where possible, used AI judges only for semantic qualities, separated CI gates from production monitoring, and closed the loop from production traces back into regression datasets."

## Implementation Order

Build this in small increments.

| Phase | Deliverable | Blocking value |
| --- | --- | --- |
| 1 | Local typed eval schema and smoke examples | Establish target behavior |
| 2 | Deterministic intent/location/time/tool evaluators | Catch routing regressions |
| 3 | Frozen weather snapshot evaluator | Catch factual regressions |
| 4 | Domain-boundary dataset and refusal evaluator | Prevent off-topic behavior |
| 5 | LangSmith experiment runner | Make results visible and comparable |
| 6 | CI smoke gate | Block obvious regressions |
| 7 | AI-judge groundedness and relevance evaluators | Score semantic quality |
| 8 | Judge calibration examples | Make AI judges trustworthy |
| 9 | Production sampling and annotation workflow | Close the real-world feedback loop |
| 10 | Pairwise baseline-vs-candidate evals | Support prompt/model iteration |

## Definition Of Done

The reference implementation is complete when:

- Every major flow has a dataset slice.
- Critical gates run in CI.
- LangSmith experiments include model, prompt, git SHA, and dataset metadata.
- Weather factuality is checked against frozen data, not judged from vibes.
- Off-topic refusal is explicitly evaluated.
- Production traces are sampled and reviewed.
- Real production failures are added to a regression dataset.
- The project has a documented threshold table and rollout process.
