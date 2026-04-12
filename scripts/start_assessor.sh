#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Curiosity — Start Assessor Daemon (Daemon 1)
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SRC_DIR="${REPO_ROOT}/src"
LOG_DIR="${HOME}/curiosity/logs"
LOG_FILE="${LOG_DIR}/assessor.log"
PID_FILE="${LOG_DIR}/assessor.pid"

# ── Ensure log directory exists ───────────────────────────────────────────────
mkdir -p "${LOG_DIR}"

# ── Guard against duplicate instances ────────────────────────────────────────
if [[ -f "${PID_FILE}" ]]; then
    OLD_PID=$(cat "${PID_FILE}")
    if kill -0 "${OLD_PID}" 2>/dev/null; then
        echo "[assessor] Already running with PID ${OLD_PID}. Exiting."
        exit 0
    else
        echo "[assessor] Stale PID file found (${OLD_PID}), removing."
        rm -f "${PID_FILE}"
    fi
fi

# ── Optional: activate virtualenv if present ──────────────────────────────────
if [[ -f "${REPO_ROOT}/venv/bin/activate" ]]; then
    # shellcheck source=/dev/null
    source "${REPO_ROOT}/venv/bin/activate"
    echo "[assessor] Activated venv at ${REPO_ROOT}/venv"
elif [[ -f "${HOME}/curiosity/venv/bin/activate" ]]; then
    # shellcheck source=/dev/null
    source "${HOME}/curiosity/venv/bin/activate"
    echo "[assessor] Activated venv at ${HOME}/curiosity/venv"
fi

# ── Environment defaults (override by exporting before running this script) ───
export REDIS_HOST="${REDIS_HOST:-localhost}"
export REDIS_PORT="${REDIS_PORT:-6379}"
export SERVER_URL="${SERVER_URL:-http://localhost:8001/v1/chat/completions}"
export SERVER_MODEL="${SERVER_MODEL:-curiosity-server}"
export GAP_SCAN_INTERVAL="${GAP_SCAN_INTERVAL:-300}"
export FAILURE_THRESHOLD="${FAILURE_THRESHOLD:-0.10}"
export CURIOSITY_NOVEL_COUNT="${CURIOSITY_NOVEL_COUNT:-3}"
export PYTHONPATH="${SRC_DIR}:${PYTHONPATH:-}"

echo "[assessor] Starting Assessor daemon…"
echo "[assessor] PYTHONPATH=${PYTHONPATH}"
echo "[assessor] Redis: ${REDIS_HOST}:${REDIS_PORT}"
echo "[assessor] Server: ${SERVER_URL} (model=${SERVER_MODEL})"
echo "[assessor] Gap scan interval: ${GAP_SCAN_INTERVAL}s | Failure threshold: ${FAILURE_THRESHOLD}"
echo "[assessor] Log: ${LOG_FILE}"

# ── Launch daemon in background ───────────────────────────────────────────────
nohup python3 -m assessor.assessor \
    >> "${LOG_FILE}" 2>&1 &

DAEMON_PID=$!
echo "${DAEMON_PID}" > "${PID_FILE}"
echo "[assessor] Daemon started with PID ${DAEMON_PID}"
echo "[assessor] Tail logs: tail -f ${LOG_FILE}"
