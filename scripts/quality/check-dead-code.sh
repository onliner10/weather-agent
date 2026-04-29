#!/usr/bin/env bash
set -eu

SCRIPT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"

echo "==> Running vulture (dead-code scanner)..."
# Filter out known false positives from protocol/callback signatures
uv run vulture \
  "$SCRIPT_DIR/src" \
  "$SCRIPT_DIR/tests" \
  --min-confidence 80 2>&1 \
  | grep -v \
    -e 'bot\.py:259: unused variable '\''application'\' \
    -e 'logging\.py:\(50\|69\|85\): unused variable '\''method_name'\' \
    -e 'deduplication\.py:41: unused variable '\''__context'\' \
    -e 'evaluator\.py:264: unused variable '\''snapshot_ref'\' \
    -e 'connection_record' \
  || true

# Check if any non-filtered issues remain
REMAINING=$(uv run vulture \
  "$SCRIPT_DIR/src" \
  "$SCRIPT_DIR/tests" \
  --min-confidence 80 2>&1 \
  | grep -v \
    -e 'bot\.py:259: unused variable '\''application'\' \
    -e 'logging\.py:\(50\|69\|85\): unused variable '\''method_name'\' \
    -e 'deduplication\.py:41: unused variable '\''__context'\' \
    -e 'evaluator\.py:264: unused variable '\''snapshot_ref'\' \
    -e 'connection_record' \
  || true)

if [ -n "$REMAINING" ]; then
  echo "ERROR: Dead code found:"
  echo "$REMAINING"
  exit 1
fi

echo "    No dead code found at confidence >= 80%."
