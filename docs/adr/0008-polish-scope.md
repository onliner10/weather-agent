# ADR 0008: Polish language and Europe/Warsaw timezone scope for MVP

## Context

Natural-language time phrases, unit expectations, and weather semantics become much more complex once the bot supports multiple locales and timezones.

## Decision

Limit the MVP to Polish-language user interactions and a default timezone of `Europe/Warsaw`, with one global unit configuration shared across the deployment.

## Consequences

- Date-resolution logic can be tightly specified and tested for Polish phrases such as `jutro`, `weekend`, and `majówka`.
- Prompting and user-facing text remain simpler and more consistent.
- Multi-language and multi-timezone support are explicit future extensions, not hidden MVP obligations.
