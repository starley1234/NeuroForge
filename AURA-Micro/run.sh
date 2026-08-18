#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
PY="${PYTHON:-python3}"
cmd="${1:-serve}"
shift || true
case "$cmd" in
  train) exec "$PY" -m aura_micro train "$@" ;;
  serve|test|demo) exec "$PY" -m aura_micro serve "$@" ;;
  infer) exec "$PY" -m aura_micro infer "$@" ;;
  download) exec "$PY" -m aura_micro download "$@" ;;
  info) exec "$PY" -m aura_micro info "$@" ;;
  *) exec "$PY" -m aura_micro "$cmd" "$@" ;;
esac
