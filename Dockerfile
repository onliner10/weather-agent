FROM python:3.12-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

COPY pyproject.toml uv.lock ./

RUN uv sync --frozen --no-dev --no-install-project

FROM python:3.12-slim

WORKDIR /app

COPY --from=builder /app/.venv /app/.venv
COPY alembic/ /app/alembic/
COPY alembic.ini /app/alembic.ini
COPY src/ /app/src/

ENV PYTHONPATH=/app/src
ENV PATH="/app/.venv/bin:$PATH"

RUN useradd --system --create-home appuser
USER appuser

HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
  CMD python -c "from weather_agent import __version__; print(__version__)" || exit 1

ENTRYPOINT ["python", "-m", "weather_agent"]
