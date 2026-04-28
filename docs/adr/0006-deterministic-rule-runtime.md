# ADR 0006: Deterministic rule evaluation with LLM-only proposal flows

## Context

The bot must explain why a notification fired and must avoid fabricating or reinterpreting weather evidence at send time.

## Decision

Use the LLM only to propose or edit rules in conversational flows. Persist only validated CEL expressions and evaluate them later with deterministic services against stored weather snapshots.

## Consequences

- Notification behavior becomes reproducible from stored data.
- Explanation flows can rely on persisted evidence instead of model memory.
- Conversational convenience is separated from runtime correctness.
