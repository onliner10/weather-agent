# ADR 0005: Split weather integrations into Forecast, Observation, and Warning providers

## Context

The MVP uses multiple public weather data sources with different payloads, coverage models, and failure modes.

## Decision

Define separate provider abstractions for forecasts, observations, and official warnings, each returning normalized contracts while preserving raw payloads for audit and debugging.

## Consequences

- Adapters can evolve independently when upstream APIs drift.
- Downstream services can consume stable domain models instead of source-specific payloads.
- Provider failures remain isolated to adapter boundaries rather than leaking into graph logic.
