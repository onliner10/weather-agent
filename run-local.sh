#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PID_FILE="${WEATHER_AGENT_PID_DIR:-/tmp}/weather-agent-bot.pid"

if [ -f "$PID_FILE" ]; then
  old_pid=$(cat "$PID_FILE")
  if kill -0 "$old_pid" 2>/dev/null; then
    echo "Killing existing bot instance (PID $old_pid)..."
    kill "$old_pid"
    for i in $(seq 1 10); do
      if ! kill -0 "$old_pid" 2>/dev/null; then
        break
      fi
      sleep 0.5
    done
    if kill -0 "$old_pid" 2>/dev/null; then
      echo "Force killing... (PID $old_pid)"
      kill -9 "$old_pid" 2>/dev/null || true
    fi
  fi
  rm -f "$PID_FILE"
fi

cd "$SCRIPT_DIR"
exec uv run python -m weather_agent bot
