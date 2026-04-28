# ADR 0002: LangGraph, LangChain, and LangSmith for learning-first orchestration

## Context

The project is intended to explore agent orchestration and observability while keeping runtime correctness anchored in deterministic weather and rule logic.

## Decision

Use LangGraph for conversational workflow orchestration, LangChain integrations where needed, and LangSmith for trace and evaluation visibility. These tools support learning and diagnostics, but they do not own correctness for date resolution, provider data, or rule evaluation.

## Consequences

- Graph traces and tool calls can be inspected without turning the system into an opaque agent.
- Deterministic services remain testable outside the orchestration stack.
- Runtime rule execution must continue to work even if tracing or LLM tooling is unavailable.
