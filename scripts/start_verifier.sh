#!/usr/bin/env bash
# =============================================================================
# Curiosity — Start Verifier Daemon (Daemon 4)
# =============================================================================
# Usage:
#   ./scripts/start_verifier.sh [--foreground]
#
# By default runs in the background and writes PID to ~/curiosity/logs/verifier.pid
# Pass --foreground (or -f) to keep it attached to the terminal.
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Resolve repo root (script lives in <repo>/scripts/)
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# ---------------------------------------------------------------------------
# Defaults (all override-able via env)
# ---------------------------------------------------------------------------
export REDIS_HOST="${REDIS_HOST:-localhost}"
export REDIS_PORT="${REDIS_PORT:-6379}"
export VLLM_BASE_URL="${VLLM_BASE_URL:-http://localhost:8001}"

export CURIOSITY_LOG_DIR="${CURIOSITY_LOG_DIR:-${HOME}/curiosity/logs}"
export CURIOSITY_CHECKPOINT_DIR="${CURIOSITY_CHECKPOINT_DIR:-${HOME}/curiosity/checkpoints}"
export CURIOSITY_BENCHMARK_DIR="${CURIOSITY_BENCHMARK_DIR:-${HOME}/curiosity/benchmarks/regression}"
export CURIOSITY_SYSTEM_PROMPT_PATH="${CURIOSITY_SYSTEM_PROMPT_PATH:-${HOME}/curiosity_code/config/system_prompt.txt}"

export SOLVE_QUEUE="${SOLVE_QUEUE:-SOLVE_QUEUE}"
export MEMORIZE_QUEUE="${MEMORIZE_QUEUE:-MEMORIZE_QUEUE}"

LOG_FILE="${CURIOSITY_LOG_DIR}/verifier.log"
PID_FILE="${CURIOSITY_LOG_DIR}/verifier.pid"

# ---------------------------------------------------------------------------
# Parse args
# ---------------------------------------------------------------------------
FOREGROUND=0
for arg in "$@"; do
  case "$arg" in
    --foreground|-f) FOREGROUND=1 ;;
    --help|-h)
      echo "Usage: $0 [--foreground|-f]"
      exit 0
      ;;
    *)
      echo "Unknown argument: $arg"
      exit 1
      ;;
  esac
done

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
mkdir -p "${CURIOSITY_LOG_DIR}" "${CURIOSITY_CHECKPOINT_DIR}" "${CURIOSITY_BENCHMARK_DIR}"

echo "[start_verifier] Repo root:    ${REPO_ROOT}"
echo "[start_verifier] Log file:     ${LOG_FILE}"
echo "[start_verifier] Redis:        ${REDIS_HOST}:${REDIS_PORT}"
echo "[start_verifier] vLLM:         ${VLLM_BASE_URL}"
echo "[start_verifier] Checkpoints:  ${CURIOSITY_CHECKPOINT_DIR}"

# ---------------------------------------------------------------------------
# Python interpreter detection
# ---------------------------------------------------------------------------
PYTHON="${PYTHON:-}"
if [[ -z "${PYTHON}" ]]; then
  for candidate in python3 python python3.11 python3.10 python3.9; do
    if command -v "${candidate}" &>/dev/null; then
      PYTHON="${candidate}"
      break
    fi
  done
fi

if [[ -z "${PYTHON}" ]]; then
  echo "[start_verifier] ERROR: No Python interpreter found. Set PYTHON env var."
  exit 1
fi
echo "[start_verifier] Python:       $("${PYTHON}" --version 2>&1)"

# ---------------------------------------------------------------------------
# Check dependencies
# ---------------------------------------------------------------------------
"${PYTHON}" -c "import redis, requests" 2>/dev/null || {
  echo "[start_verifier] WARNING: redis or requests not importable — attempting install"
  "${PYTHON}" -m pip install redis requests --quiet
}

# ---------------------------------------------------------------------------
# Kill existing daemon if running
# ---------------------------------------------------------------------------
if [[ -f "${PID_FILE}" ]]; then
  OLD_PID="$(cat "${PID_FILE}")"
  if kill -0 "${OLD_PID}" 2>/dev/null; then
    echo "[start_verifier] Stopping existing daemon (pid=${OLD_PID})"
    kill "${OLD_PID}" && sleep 1
  fi
  rm -f "${PID_FILE}"
fi

# ---------------------------------------------------------------------------
# Launch
# ---------------------------------------------------------------------------
CMD=("${PYTHON}" -m src.verifier.verifier)

if [[ "${FOREGROUND}" -eq 1 ]]; then
  echo "[start_verifier] Running in foreground (Ctrl-C to stop)"
  cd "${REPO_ROOT}"
  exec "${CMD[@]}"
else
  echo "[start_verifier] Launching daemon in background..."
  cd "${REPO_ROOT}"
  nohup "${CMD[@]}" >> "${LOG_FILE}" 2>&1 &
  DAEMON_PID=$!
  echo "${DAEMON_PID}" > "${PID_FILE}"
  echo "[start_verifier] Verifier daemon started (pid=${DAEMON_PID})"
  echo "[start_verifier] Logs: tail -f ${LOG_FILE}"
fi
