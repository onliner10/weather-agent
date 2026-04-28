# ADR 0007: Docker Compose as the initial homelab deployment target

## Context

The first production-like environment is a single self-hosted homelab setup rather than a distributed cloud platform.

## Decision

Standardize the MVP runtime around Docker Compose for local development and the first deployment target, including application services, workers, and PostgreSQL/TimescaleDB.

## Consequences

- Local and deployment workflows share the same basic topology.
- Operational guidance can focus on health checks, secrets, persistence, and scheduled jobs instead of cluster concerns.
- Horizontal scaling and more advanced orchestration are deferred until after MVP validation.
