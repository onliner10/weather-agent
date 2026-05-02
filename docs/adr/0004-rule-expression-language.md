# ADR 0004: CEL rule expressions with allowlisted weather helpers

## Context

Users need expressive weather notification logic, but evaluating arbitrary Python or model-generated code at runtime would be unsafe and non-deterministic.

## Decision

Represent rule expressions as CEL and validate them against an explicit allowlist of helper functions and weather metrics before a rule can be activated.

## Consequences

- Rule storage and evaluation stay deterministic and inspectable.
- LLMs can propose expressions, but activation requires validation and user confirmation.
- New metrics or helper functions must be explicitly added to the allowlist and tested.
