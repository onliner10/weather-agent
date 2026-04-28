#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

if [ ! -f "${PROJECT_DIR}/.env" ]; then
  echo "ERROR: .env file not found. Copy .env.example to .env and fill in values." >&2
  exit 1
fi

cd "${PROJECT_DIR}"

exec python -m weather_agent bot