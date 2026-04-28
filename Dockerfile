FROM python:3.12-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

COPY pyproject.toml uv.lock ./

RUN uv sync --frozen --no-dev

FROM python:3.12-slim

WORKDIR /app

COPY --from=builder /app/.venv /app/.venv
COPY src/ /app/src/

ENV PYTHONPATH=/app/src
ENV PATH="/app/.venv/bin:$PATH"

RUN useradd --system --create-home appuser
USER appuser

ENTRYPOINT ["python", "-m", "weather_agent"]