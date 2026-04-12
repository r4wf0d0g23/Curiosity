#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Curiosity — Start Memorizer Daemon (Daemon 5)
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SRC_DIR="${REPO_ROOT}/src"
LOG_DIR="${HOME}/curiosity/logs"
LOG_FILE="${LOG_DIR}/memorizer.log"
PID_FILE="${LOG_DIR}/memorizer.pid"

# ── Ensure log directory exists ───────────────────────────────────────────────
mkdir -p "${LOG_DIR}"

# ── Guard against duplicate instances ────────────────────────────────────────
if [[ -f "${PID_FILE}" ]]; then
    OLD_PID=$(cat "${PID_FILE}")
    if kill -0 "${OLD_PID}" 2>/dev/null; then
        echo "[memorizer] Already running with PID ${OLD_PID}. Exiting."
        exit 0
    else
        echo "[memorizer] Stale PID file found (${OLD_PID}), removing."
        rm -f "${PID_FILE}"
    fi
fi

# ── Optional: activate virtualenv if present ──────────────────────────────────
if [[ -f "${REPO_ROOT}/venv/bin/activate" ]]; then
    # shellcheck source=/dev/null
    source "${REPO_ROOT}/venv/bin/activate"
    echo "[memorizer] Activated venv at ${REPO_ROOT}/venv"
elif [[ -f "${HOME}/curiosity/venv/bin/activate" ]]; then
    # shellcheck source=/dev/null
    source "${HOME}/curiosity/venv/bin/activate"
    echo "[memorizer] Activated venv at ${HOME}/curiosity/venv"
fi

# ── Environment defaults (override by exporting before running this script) ───
export REDIS_HOST="${REDIS_HOST:-localhost}"
export REDIS_PORT="${REDIS_PORT:-6379}"
export IMPROVEMENT_LOOP_INTERVAL="${IMPROVEMENT_LOOP_INTERVAL:-20}"
export PYTHONPATH="${SRC_DIR}:${PYTHONPATH:-}"

echo "[memorizer] Starting Memorizer daemon…"
echo "[memorizer] PYTHONPATH=${PYTHONPATH}"
echo "[memorizer] Redis: ${REDIS_HOST}:${REDIS_PORT}"
echo "[memorizer] Log: ${LOG_FILE}"

# ── Launch daemon in background ───────────────────────────────────────────────
nohup python3 -m memorizer.memorizer \
    >> "${LOG_FILE}" 2>&1 &

DAEMON_PID=$!
echo "${DAEMON_PID}" > "${PID_FILE}"
echo "[memorizer] Daemon started with PID ${DAEMON_PID}"
echo "[memorizer] Tail logs: tail -f ${LOG_FILE}"
