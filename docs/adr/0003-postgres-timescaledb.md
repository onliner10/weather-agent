# ADR 0003: PostgreSQL with TimescaleDB for relational and time-series persistence

## Context

The MVP must store user configuration, rules, audit history, weather snapshots, and explanation-ready evaluation evidence.

## Decision

Use PostgreSQL as the primary database and TimescaleDB extensions where time-series storage improves forecast, observation, warning, and evaluation persistence.

## Consequences

- One database stack supports both transactional and historical workloads.
- Migration discipline becomes part of the core delivery path.
- Local development and homelab deployment depend on a database service being available.
