# Implement Epic: Telegram Weather AI Agent

## Metadata
```yaml
id: implement-epic-20260428t103903z-c27038d9
spec: docs/specs/implement-epic-20260428T103903Z-c27038d9.md
owner: Mateusz Urban
status: draft
created: 2026-04-28
updated: 2026-04-28
```

## Phases
- `Phase 1 - Foundation and contracts`: establish the Python/uv skeleton, typed settings, ADRs, domain vocabulary, and agent workflow guidance. Assumption: the repository is still effectively greenfield and can adopt the `src/weather_agent/` package layout. Maps `WA-001`, `WA-002`, `WA-003`, `WA-035`.
- `Phase 2 - Deployment and persistence base`: create the Docker Compose baseline and Alembic-backed PostgreSQL/TimescaleDB schema required by all later slices. Maps `WA-004`, `WA-005`.
- `Phase 3 - Core domain services`: implement authorization, global settings, named locations, deterministic Polish date resolution, and provider-facing normalized weather contracts. Maps `WA-006`, `WA-007`, `WA-008`, `WA-009`.
- `Phase 4 - Weather ingestion and storage`: add Open-Meteo forecast, IMGW synop, IMGW warnings, persistence repositories, and forecast-comparison utilities. Maps `WA-010`, `WA-011`, `WA-012`, `WA-013`, `WA-014`.
- `Phase 5 - Rule runtime`: integrate CEL validation/evaluation, rule CRUD with short IDs, evaluation worker, cooldown/snooze logic, and explanation-ready notification evidence. Maps `WA-015`, `WA-016`, `WA-017`, `WA-018`, `WA-019`.
- `Phase 6 - Telegram channel`: build the Telegram ingress, thread/topic context mapping, deterministic commands, and outbound notification sender. Maps `WA-020`, `WA-021`, `WA-022`, `WA-023`.
- `Phase 7 - LangGraph orchestration and observability`: wire model selection, typed conversational state, weather/rule flows, thread-scoped memory, LangSmith tracing, structured logging, status surfaces, cleanup jobs, and production-like Compose finishing work. Maps `WA-024`, `WA-025`, `WA-026`, `WA-027`, `WA-028`, `WA-029`, `WA-031`, `WA-032`, `WA-033`, `WA-034`.
- `Phase 8 - Verification and release`: add eval datasets, mocked end-to-end coverage, real-provider smoke checks, and final deployment/release documentation. Maps `WA-030`, `WA-036`, `WA-037`, `WA-038`.

## Tasks (machine-readable)
- `docs/plans/implement-epic-20260428T103903Z-c27038d9-tasks.json` must conform to `docs/contracts/autonomy-tasks.schema.json`.

## Task list (human summary)
| id | title | deps | status | notes |
| --- | --- | --- | --- | --- |
| phase-01 | Foundation and contracts | N/A | todo | Covers `WA-001`, `WA-002`, `WA-003`, `WA-035`; assumes no existing app skeleton must be preserved. |
| phase-02 | Deployment and persistence base | phase-01 | todo | Covers `WA-004`, `WA-005`; unlocks DB-backed services and local stack bootstrapping. |
| phase-03 | Core domain services | phase-01, phase-02 | todo | Covers `WA-006`, `WA-007`, `WA-008`, `WA-009`; assumes Polish-only and `Europe/Warsaw` remain MVP scope boundaries. |
| phase-04 | Weather ingestion and storage | phase-01, phase-02, phase-03 | todo | Covers `WA-010` to `WA-014`; assumes Open-Meteo DWD ICON and IMGW remain the only MVP providers. |
| phase-05 | Rule runtime | phase-03, phase-04 | todo | Covers `WA-015` to `WA-019`; preserves the hard constraint that runtime rule evaluation never calls the LLM. |
| phase-06 | Telegram channel | phase-02, phase-03, phase-05 | todo | Covers `WA-020` to `WA-023`; assumes Telegram private supergroup plus Topics remains the preferred operating mode. |
| phase-07 | LangGraph orchestration and observability | phase-03, phase-04, phase-05, phase-06 | todo | Covers `WA-024` to `WA-034` except eval/release tasks; keeps LangSmith optional for correctness but required for learning visibility. |
| phase-08 | Verification and release | phase-04, phase-05, phase-06, phase-07 | todo | Covers `WA-030`, `WA-036`, `WA-037`, `WA-038`; exit gate for MVP readiness. |

## Risks
- Provider volatility risk: Open-Meteo or IMGW payload drift can break adapters and downstream persistence assumptions.
- Scope-coupling risk: the backlog is dependency-rich, so weak contract discipline in early phases will create rework in LangGraph, Telegram, and rule execution.
- Natural-language ambiguity risk: Polish date phrases such as `weekend` and `majówka` can create user-visible surprises unless resolver behavior is frozen and documented early.
- Operational risk: homelab deployment depends on correct secrets, persistent volumes, clock correctness, and scheduler behavior that may not surface until late integration.
- Assumption risk: this plan assumes the current `weather-agent-778` child backlog is complete enough to drive MVP execution without creating major new epics.

## Evidence checklist
- `docs/specs/implement-epic-20260428T103903Z-c27038d9.md` remains the narrative source of truth for goals, constraints, and acceptance.
- `docs/plans/implement-epic-20260428T103903Z-c27038d9-tasks.json` stays in sync with the phase breakdown and references concrete file targets.
- Beads dependency graph for `weather-agent-778` remains the detailed execution ordering beneath this plan.
- Each completed phase has passing tests or smoke checks tied to its child Beads tasks before downstream phases are considered complete.
- Rule-runtime work proves deterministic CEL validation/evaluation without LLM calls in worker paths.
- Final verification includes one mocked end-to-end MVP path and one real-provider smoke pass for Open-Meteo and IMGW before release documentation is treated as done.

## Rollout / rollback
- Roll out in phase order, but use the existing Beads child dependencies as the finer-grained scheduler within each phase.
- Do not advance Telegram, LangGraph, or release work by bypassing unresolved persistence or contract failures; treat those as rollback points to the previous stable phase.
- If provider integrations or rule evaluation destabilize the stack, fall back to the last passing boundary with working typed contracts, persistence migrations, and mocked tests before resuming.
- Keep Docker Compose, migrations, and environment contracts backward-compatible across adjacent phases where practical so local rollback remains a matter of reverting the latest slice rather than rebuilding the project skeleton.
