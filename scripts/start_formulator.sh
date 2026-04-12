#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# Curiosity — start_formulator.sh
# Launches Daemon 2: FORMULATOR
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
SRC_DIR="$REPO_ROOT/src"
LOG_DIR="${HOME}/curiosity/logs"

# ── Environment defaults (override via env vars) ─────────────────────────────
export REDIS_HOST="${REDIS_HOST:-localhost}"
export REDIS_PORT="${REDIS_PORT:-6379}"
export VLLM_HOST="${VLLM_HOST:-localhost}"
export VLLM_PORT="${VLLM_PORT:-8001}"
export VLLM_MODEL="${VLLM_MODEL:-curiosity-server}"

# ── Logging directory ─────────────────────────────────────────────────────────
mkdir -p "$LOG_DIR"

echo "=========================================="
echo "  Curiosity FORMULATOR daemon"
echo "  Redis  : ${REDIS_HOST}:${REDIS_PORT}"
echo "  vLLM   : ${VLLM_HOST}:${VLLM_PORT} (${VLLM_MODEL})"
echo "  Logs   : ${LOG_DIR}/formulator.log"
echo "=========================================="

# ── Launch ────────────────────────────────────────────────────────────────────
exec python3 -m formulator.formulator "$@"
