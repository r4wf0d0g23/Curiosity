#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Curiosity — Start Solver Daemon (Daemon 3)
# Memory-first retrieval + novel solution generation via vLLM Server
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SRC_DIR="${REPO_ROOT}/src"
LOG_DIR="${HOME}/curiosity/logs"
LOG_FILE="${LOG_DIR}/solver.log"
PID_FILE="${LOG_DIR}/solver.pid"

# ── Ensure log directory exists ───────────────────────────────────────────────
mkdir -p "${LOG_DIR}"

# ── Guard against duplicate instances ────────────────────────────────────────
if [[ -f "${PID_FILE}" ]]; then
    OLD_PID=$(cat "${PID_FILE}")
    if kill -0 "${OLD_PID}" 2>/dev/null; then
        echo "[solver] Already running with PID ${OLD_PID}. Exiting."
        exit 0
    else
        echo "[solver] Stale PID file found (${OLD_PID}), removing."
        rm -f "${PID_FILE}"
    fi
fi

# ── Optional: activate virtualenv if present ─────────────────────────────────
if [[ -f "${REPO_ROOT}/venv/bin/activate" ]]; then
    # shellcheck source=/dev/null
    source "${REPO_ROOT}/venv/bin/activate"
    echo "[solver] Activated venv at ${REPO_ROOT}/venv"
elif [[ -f "${HOME}/curiosity/venv/bin/activate" ]]; then
    # shellcheck source=/dev/null
    source "${HOME}/curiosity/venv/bin/activate"
    echo "[solver] Activated venv at ${HOME}/curiosity/venv"
fi

# ── Environment defaults (override by exporting before running this script) ───
export REDIS_HOST="${REDIS_HOST:-localhost}"
export REDIS_PORT="${REDIS_PORT:-6379}"
export PYTHONPATH="${SRC_DIR}:${PYTHONPATH:-}"

echo "[solver] Starting Solver daemon…"
echo "[solver] PYTHONPATH=${PYTHONPATH}"
echo "[solver] Redis:     ${REDIS_HOST}:${REDIS_PORT}"
echo "[solver] ChromaDB:  localhost:8000"
echo "[solver] vLLM:      localhost:8001"
echo "[solver] Embed:     localhost:8004"
echo "[solver] Log:       ${LOG_FILE}"

# ── Launch daemon in background ───────────────────────────────────────────────
nohup python3 -m solver.solver \
    >> "${LOG_FILE}" 2>&1 &

DAEMON_PID=$!
echo "${DAEMON_PID}" > "${PID_FILE}"
echo "[solver] Daemon started with PID ${DAEMON_PID}"
echo "[solver] Tail logs: tail -f ${LOG_FILE}"
