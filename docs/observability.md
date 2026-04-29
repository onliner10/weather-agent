# Observability

## Overview

This project uses three complementary observability pillars:

| Pillar | Backend | Purpose |
|--------|---------|---------|
| **Structured logs** | Grafana Cloud Loki | Operational debugging, error tracking, audit trail |
| **Application metrics** | Grafana Cloud Prometheus | Service health, throughput, latency, failure rates |
| **Traces** | LangSmith | Conversational turn detail, LLM calls, graph execution |

**How they complement each other:**

- **Logs** tell you _what happened_ — an error occurred, a message was received, a worker cycle completed. Each log line carries a `correlation_id` so you can follow a single request across components.
- **Metrics** tell you _how the system is behaving over time_ — request rates, error ratios, latency distributions, staleness gauges. Metrics drive dashboards and alerts.
- **Traces** tell you _the detailed path through the graph_ — which nodes ran, which LLM calls were made, how long each step took. Traces are linked to logs via `correlation_id` in metadata.

Debugging workflow: when a metric alert fires (e.g. conversation error spike), open the relevant dashboard to see the time window, then switch to Loki and search `{service="bot"} |= "error"` filtered by that time range, then correlate with LangSmith traces using the `correlation_id` from the log entry.

## Component Architecture

```
                    ┌──────────────────────────────────────────────┐
                    │              Grafana Cloud                   │
                    │  ┌─────────┐  ┌──────────┐                   │
                    │  │Prometheus│  │  Loki    │                   │
                    │  └────▲─────┘  └────▲─────┘                   │
                    │       │              │                         │
                    └───────┼──────────────┼─────────────────────────┘
                            │              │
                    ┌───────┴──────────────┴───────┐
                    │       Grafana Alloy           │
                    │  ┌──────────┐ ┌───────────┐   │
                    │  │ Scrape   │ │Container  │   │
                    │  │ metrics  │ │logs       │   │
                    │  └────┬─────┘ └─────┬─────┘   │
                    └───────┼──────────────┼─────────┘
                            │              │
        ┌───────────────────┼──────────────┼───────────────────┐
        │                   │              │                    │
┌───────▼───────┐   ┌──────▼──────┐  ┌────▼──────────┐        │
│weather-agent- │   │weather-agent│  │postgres-      │        │
│bot :8080      │   │-worker:8081 │  │exporter :9187 │        │
│/health        │   │/health      │  │/metrics       │        │
│/metrics       │   │/metrics     │  └───────────────┘        │
└───────▲───────┘   └──────▲───────┘                         │
        │                   │                                  │
        └───────────────────┼──────────────────────────────────┘
                            │
                    ┌───────▼───────┐
                    │ postgres-     │
                    │ timescaledb   │
                    └───────────────┘
```

### Bot (`weather-agent-bot`)

- **Logs**: Structured JSON to stdout. Each Telegram message handler call is wrapped in `bound_telegram_context` which binds `correlation_id`, `chat_id`, `message_thread_id`, `telegram_user_id`, `message_id`, `service="bot"`, `component="telegram_handler"`.
- **Metrics** (port `8080`): `/metrics` exposes 16 metrics covering Telegram messages, conversation turns, reply sends, tool calls, LLM requests, geocode requests, and provider requests.
- **Traces**: LangSmith is called via `configure_tracing()` in `__main__.py`. Each conversation turn creates a `telegram-turn:<context_key>:<intent>` trace.
- **Health**: `/health` returns JSON with db status, last forecast fetch, last rule evaluation, LangSmith status.

### Worker (`weather-agent-worker`)

- **Logs**: Structured JSON to stdout. Each worker cycle is wrapped in `bound_worker_context` which binds `correlation_id`, `service="worker"`, `component="rule_evaluator"`.
- **Metrics** (port `8081`): `/metrics` exposes 10 metrics covering worker cycles, rule evaluations, notifications, and forecast refreshes, plus two last-success timestamps as gauges.
- **Traces**: Worker cycles create a `worker_cycle` trace with `evaluate_rules`, `evaluate_single_rule`, `forecast_refresh`, and `evaluation_result` subtraces.
- **Health**: `/health` (same schema as bot).

### Postgres exporter (`postgres-exporter`)

- Runs `prometheuscommunity/postgres-exporter:latest`.
- Exposes Postgres metrics at `:9187/metrics` on the internal network.
- Connects via `DATA_SOURCE_NAMNe` DSN from `POSTGRES_EXPORTER_DSN` env var.
- Provides: connection counts, transaction rates, cache hit ratios, database size, replication stats.

### Grafana Alloy

- Runs `grafana/alloy:latest`.
- Reads config from `ops/alloy/config.alloy` (River format).
- **Log collection**: `loki.source.docker` reads Docker container logs via the Docker socket, labels them with `container` and `stream`, and ships to Grafana Cloud Loki.
- **Metrics scraping**: Three `prometheus.scrape` blocks:
  - `weather_agent_bot` → `weather-agent-bot:8080/metrics`
  - `weather_agent_worker` → `weather-agent-worker:8081/metrics`
  - `postgres_exporter` → `postgres-exporter:9187/metrics`
- **Remote write**: `loki.write.grafana_cloud` and `prometheus.remote_write.grafana_cloud` with basic auth using `GRAFANA_CLOUD_*` credentials.

### Intentionally logged metadata

Contextual fields bound per-message or per-cycle:
- `correlation_id` (UUID v4, links traces to logs)
- `service` (`bot` | `worker`)
- `component` (`telegram_handler` | `rule_evaluator`)
- `chat_id`, `message_thread_id`, `telegram_user_id`, `message_id`, `reply_to_message_id`
- `context_key`
- `event`, `level`, `logger_name`, `timestamp` (ISO 8601 UTC)

### Intentionally excluded from logs

- **Raw user message bodies** — not logged by default. Message text is excluded from structured log fields.
- **Secrets** — `_redact_secrets` processor masks keys matching patterns like `api_key`, `token`, `password`, `authorization`, `secret`, `cookie`, `session`, `private_key`, `database_url`. Additionally, values matching Telegram token format (`123456:ABCdef`), OpenAI-style keys (`sk-...`), Bearer tokens, and Basic auth are redacted.
- **Full provider payloads** — raw forecast/observation JSON is not logged.
- **Full LLM prompts/responses** — not written to log entries.
- **`recent_context` (turn history)** — excluded from traces and logs.

## Grafana Cloud Setup

### Required environment variables

These vars are consumed by Grafana Alloy and defined in `.env.example`:

| Variable | Used by | Description |
|---|---|---|
| `GRAFANA_CLOUD_PROM_URL` | Alloy `prometheus.remote_write` | Grafana Cloud Prometheus remote write endpoint (e.g. `https://prometheus-prod-XX.grafana.net/api/prom/push`) |
| `GRAFANA_CLOUD_PROM_USER` | Alloy `prometheus.remote_write` | Grafana Cloud Prometheus instance ID (numeric) |
| `GRAFANA_CLOUD_PROM_API_KEY` | Alloy `prometheus.remote_write` | Grafana Cloud API key with `metrics:write` scope |
| `GRAFANA_CLOUD_LOKI_URL` | Alloy `loki.write` | Grafana Cloud Loki push endpoint (e.g. `https://logs-prod-XX.grafana.net/loki/api/v1/push`) |
| `GRAFANA_CLOUD_LOKI_USER` | Alloy `loki.write` | Grafana Cloud Loki instance ID (numeric) |
| `GRAFANA_CLOUD_LOKI_API_KEY` | Alloy `loki.write` | Grafana Cloud API key with `logs:write` scope |
| `POSTGRES_EXPORTER_DSN` | `postgres-exporter` container | Postgres DSN for exporter (e.g. `postgresql://weather_agent:weather_agent@postgres-timescaledb:5432/weather_agent`) |

### Step-by-step setup

1. **Create a Grafana Cloud account** at [grafana.com](https://grafana.com). The free tier includes 10k series for Prometheus and 50GB of Loki logs.

2. **Get your Prometheus credentials:**
   - Go to Grafana Cloud → Your Stack → Details → Prometheus.
   - Note the instance ID (numeric) and the remote write endpoint URL.
   - Generate an API key with `metrics:write` scope: Stack → API Keys → Add API Key.

3. **Get your Loki credentials:**
   - Go to Grafana Cloud → Your Stack → Details → Loki.
   - Note the instance ID (numeric) and the push endpoint URL.
   - Generate an API key with `logs:write` scope (can reuse the same key).

4. **Configure locally:**
   ```bash
   cp .env.example .env
   # Edit .env and fill in:
   #   GRAFANA_CLOUD_PROM_URL
   #   GRAFANA_CLOUD_PROM_USER
   #   GRAFANA_CLOUD_PROM_API_KEY
   #   GRAFANA_CLOUD_LOKI_URL
   #   GRAFANA_CLOUD_LOKI_USER
   #   GRAFANA_CLOUD_LOKI_API_KEY
   ```

5. **Start the stack:**
   ```bash
   docker compose up -d
   ```

6. **Verify Alloy is running:**
   ```bash
   docker compose ps       # Should show grafana-alloy Up
   docker compose logs grafana-alloy  # Check for connection errors
   ```

## Verification Steps

### Check Alloy is scraping

```bash
# Alloy exposes a UI on port 12345 (internal network only)
# Port-forward or use docker exec:
docker compose exec grafana-alloy alloy components
# Or check Alloy logs for scrape successes:
docker compose logs grafana-alloy | grep -i "scrape\|metrics\|loki"
```

### Verify metrics in Grafana Cloud Prometheus

1. Open Grafana Cloud → Explore.
2. Select Prometheus data source.
3. Run queries:

```promql
# Check bot metrics are arriving
weather_agent_telegram_messages_total

# Check worker metrics are arriving
weather_agent_worker_cycles_total

# Check Postgres metrics are arriving
pg_stat_database_numbackends

# Check all metric names from this project
{__name__=~"weather_agent_.*"}
```

If metrics appear with values, the pipeline is working.

### Verify logs in Grafana Cloud Loki

1. Open Grafana Cloud → Explore.
2. Select Loki data source.
3. Run queries:

```logql
# All bot logs
{container="weather-agent-bot"} |= ``

# All worker logs
{container="weather-agent-worker"} |= ``

# Errors only
{container=~"weather-agent-bot|weather-agent-worker"} |= "error" |= ``

# Filter by correlation_id
{container="weather-agent-bot"} |= "correlation_id" |= "your-uuid-here"

# Recent logs (last 15 minutes)
{container="weather-agent-bot"} |= `` | logfmt | __timestamp__ > now() - 15m
```

If log lines appear, the log pipeline is working.

### Verify LangSmith traces

See [LangSmith tracing](#langsmith-integration-cross-reference) and `README.md` for detailed trace verification.

## Recommended Dashboards

### Bot overview panel

| Panel | Metric | Query |
|-------|--------|-------|
| Messages received | Counter | `rate(weather_agent_telegram_messages_total[5m])` |
| Conversation turns | Counter | `rate(weather_agent_conversation_turns_total[5m])` |
| Turn failures | Counter | `rate(weather_agent_conversation_failures_total[5m])` |
| Turn latency (p50/p95/p99) | Histogram | `histogram_quantile(0.95, rate(weather_agent_conversation_turn_duration_seconds_bucket[5m]))` |
| Authorization failures | Counter | `rate(weather_agent_authorization_failures_total[5m])` |

### Worker overview panel

| Panel | Metric | Query |
|-------|--------|-------|
| Worker cycles | Counter | `rate(weather_agent_worker_cycles_total[5m])` |
| Rules evaluated | Counter | `rate(weather_agent_rules_evaluated_total[5m])` |
| Rule failures | Counter | `rate(weather_agent_rule_evaluation_failures_total[5m])` |
| Notifications sent | Counter | `rate(weather_agent_notifications_total[5m])` |
| Forecast refreshes | Counter | `rate(weather_agent_forecast_refresh_total[5m])` |
| Cycle latency (p50/p95) | Histogram | `histogram_quantile(0.95, rate(weather_agent_worker_cycle_duration_seconds_bucket[5m]))` |
| Last successful cycle | Gauge | `weather_agent_last_successful_worker_cycle_timestamp_seconds` |
| Last forecast refresh | Gauge | `weather_agent_last_successful_forecast_refresh_timestamp_seconds` |

### Provider / error overview panel

| Panel | Metric | Query |
|-------|--------|-------|
| Provider requests | Counter | `rate(weather_agent_provider_requests_total[5m])` |
| Provider failures | Counter | `rate(weather_agent_provider_requests_total{outcome="failure"}[5m])` |
| Provider latency (p95) | Histogram | `histogram_quantile(0.95, rate(weather_agent_provider_request_duration_seconds_bucket[5m]))` |
| Provider failure rate | Ratio | `sum(rate(weather_agent_provider_requests_total{outcome="failure"}[5m])) / sum(rate(weather_agent_provider_requests_total[5m]))` |
| Tool errors | Counter | `rate(weather_agent_tool_calls_total[5m])` |
| LLM request failures | Counter | `rate(weather_agent_llm_requests_total{outcome="failure"}[5m])` |
| Geocode failures | Counter | `rate(weather_agent_geocode_requests_total{outcome="failure"}[5m])` |

### Notification / rule-evaluation panel

| Panel | Metric | Query |
|-------|--------|-------|
| Notifications by type | Counter | `rate(weather_agent_notifications_total[5m])` |
| Notification failures | Counter | `rate(weather_agent_notification_failures_total[5m])` |
| Notification send latency (p95) | Histogram | `histogram_quantile(0.95, rate(weather_agent_notification_send_duration_seconds_bucket[5m]))` |
| Rules by outcome | Counter | `rate(weather_agent_rules_evaluated_total[5m])` |
| Rule evaluation latency (p95) | Histogram | `histogram_quantile(0.95, rate(weather_agent_rule_evaluation_duration_seconds_bucket[5m]))` |

### Postgres overview panel

| Panel | Metric | Query |
|-------|--------|-------|
| Active connections | Gauge | `pg_stat_database_numbackends{datname="weather_agent"}` |
| Transactions per second | Counter | `rate(pg_stat_database_xact_commit{datname="weather_agent"}[5m])` |
| Cache hit ratio | Ratio | `rate(pg_stat_database_blks_hit{datname="weather_agent"}[5m]) / (rate(pg_stat_database_blks_hit{datname="weather_agent"}[5m]) + rate(pg_stat_database_blks_read{datname="weather_agent"}[5m]))` |
| Database size | Gauge | `pg_database_size_bytes{datname="weather_agent"}` |
| Deadlocks | Counter | `rate(pg_stat_database_deadlocks{datname="weather_agent"}[5m])` |
| Postgres exporter up | Gauge | `up{job="postgres-exporter"}` |

## Recommended Alerts

All alerts assume a standard Prometheus evaluation interval. Adjust thresholds to match your traffic and scaling.

### Conversation error spike

```promql
rate(weather_agent_conversation_failures_total[5m]) > 0.1
```

Fires when the conversation turn failure rate exceeds 0.1 errors/sec over 5 minutes. This could indicate a broken provider, a bug in the graph, or a Telegram API issue.

### Telegram send failures

```promql
rate(weather_agent_notification_failures_total[5m]) > 0
```

Fires when any notification (reply or rule-triggered) fails to send via Telegram. Any sustained rate above zero warrants investigation.

### Worker stale

```promql
time() - weather_agent_last_successful_worker_cycle_timestamp_seconds > 2 * 900
```

Fires when the worker has not completed a successful cycle for more than 2× the configured interval (default: 15 min = 900s, so > 30 min). This indicates the worker process may be stuck, crashed, or unable to connect to the DB.

### Forecast refresh stale

```promql
time() - weather_agent_last_successful_forecast_refresh_timestamp_seconds > 2 * 1800
```

Fires when no forecast refresh has succeeded for more than 2× the configured interval (default: 30 min = 1800s, so > 60 min).

### Postgres down

```promql
absent(up{job="postgres-exporter"})
```

Fires when the Postgres exporter target disappears from Prometheus service discovery. This could mean the Postgres exporter container is down or unreachable.

### Metrics absence

```promql
absent(weather_agent_conversation_turns_total)
```

Fires when the `weather_agent_conversation_turns_total` metric is absent (no data points). This indicates the bot process is not exposing metrics, or Alloy is unable to scrape it.

### Postgres health check

```promql
pg_stat_database_numbackends{datname="weather_agent"} == 0
```

Fires if the application database has no connections (all app processes disconnected).

## LangSmith Integration (cross-reference)

LangSmith is the dedicated platform for **conversational and model traces**. It is separate from Grafana Cloud and serves a different purpose.

### What LangSmith provides

- Trace hierarchy for each `telegram-turn` and `worker_cycle`
- LLM call details: prompt, response, token usage, latency
- Graph node execution timing
- Full trace search via the LangSmith UI

### Correlating traces with logs

Every LangSmith trace carries a `correlation_id` in its metadata (set via `bound_telegram_context` or `bound_worker_context`). This same `correlation_id` appears in log entries emitted during that request.

**To correlate:**

1. Find the `correlation_id` from a LangSmith trace (check the trace metadata panel).
2. Search Loki for that ID: `{container="weather-agent-bot"} |= "correlation_id=<uuid>"`
3. The log lines show you what the process logged during that trace.

### When to use what

| Situation | Use |
|-----------|-----|
| "Why did this specific conversation fail?" | LangSmith trace + Loki logs by correlation_id |
| "Are errors increasing across all users?" | Grafana Cloud Prometheus (metrics) |
| "What error message did the worker log at 03:00?" | Grafana Cloud Loki |
| "How long are LLM calls taking on average?" | Grafana Cloud Prometheus (`rate(weather_agent_llm_request_duration_seconds_bucket)`) |
| "Which nodes ran for this particular turn?" | LangSmith trace |
| "Is the worker still running?" | Grafana Cloud Prometheus (`weather_agent_last_successful_worker_cycle_timestamp_seconds`) |

### Detailed trace documentation

See `README.md` → [LangSmith tracing](../README.md#langsmith-tracing) for the full trace hierarchy, excluded metadata, and troubleshooting steps.

## Grafana Cloud Dashboard JSON

The recommended panels above can be assembled into a Grafana Cloud dashboard. The fastest approach is to create a new dashboard in Grafana Cloud, add panels using the queries above, and configure the Prometheus and Loki data sources (both are available automatically in Grafana Cloud).

For provisioning via JSON (requires Grafana API access), each panel follows this pattern:

```json
{
  "datasource": { "type": "prometheus", "uid": "grafanacloud-prom" },
  "fieldConfig": { "defaults": { "unit": "cps" }, "overrides": [] },
  "targets": [{
    "expr": "rate(weather_agent_conversation_turns_total[5m])",
    "legendFormat": "turns",
    "refId": "A"
  }],
  "title": "Conversation turns",
  "type": "timeseries"
}
```
