#!/usr/bin/env bash
# Run minimax h3 image studio on its own under `helm dev`, which keeps what it stores in ./.helm.
#
#   [MINIMAX_MODELS=<dir holding MiniMax-H3-Image and MiniMax-H3-Image-PE>] [HELM=<helm>] bash web/run.sh [helm dev flags]
#   bash web/run.sh stop
#
# A run first stops the helm dev this script started before, so starting again is a restart.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

HELM="${HELM:-helm}"
MODELS="${MINIMAX_MODELS:-$HOME/models}"
RUN="$PWD/.cache/minimax-studio"
PIDFILE="$RUN/minimax-studio.pid"

# Ends the helm dev this script last started, which stops the studio before it exits.
stop() {
  local pid
  pid="$(cat "$PIDFILE" 2>/dev/null || true)"
  if [ -n "$pid" ] && ps -p "$pid" -o command= | grep -q " dev -f helmstudio.yaml"; then
    echo "helm dev: stopping the minimax h3 image studio started before ($pid)"
    kill -TERM "$pid" 2>/dev/null || true
    while kill -0 "$pid" 2>/dev/null; do sleep 0.2; done
  fi
  rm -f "$PIDFILE"
}
if [ "${1:-}" = "stop" ]; then
  stop
  exit 0
fi

if ! command -v "$HELM" >/dev/null; then
  echo "helm is not installed. Install it with helmstudio's installer, or set HELM to its path." >&2
  exit 1
fi
if ! .venv/bin/python -c "import helm_runtime_sdk" 2>/dev/null; then
  echo "helm-runtime-sdk is not installed in .venv: run uv sync" >&2
  exit 1
fi

stop
mkdir -p "$RUN"
echo $$ >"$PIDFILE" # exec keeps this pid, so it is helm dev's
exec "$HELM" dev -f helmstudio.yaml -venv .venv \
  -link "minimax_h3_image=$MODELS/MiniMax-H3-Image" \
  -link "pe=$MODELS/MiniMax-H3-Image-PE" \
  "$@"
