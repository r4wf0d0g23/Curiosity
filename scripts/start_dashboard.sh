#!/bin/bash
# Curiosity Dashboard — launch script
set -e

REPO_DIR="/home/rawdata/curiosity_code"
LOG_FILE="$HOME/curiosity/logs/dashboard.log"
PID_FILE="$HOME/curiosity/dashboard.pid"

mkdir -p "$(dirname "$LOG_FILE")"

cd "$REPO_DIR"
pip install fastapi uvicorn --break-system-packages -q

# Stop any existing dashboard process
if [ -f "$PID_FILE" ]; then
  OLD_PID=$(cat "$PID_FILE")
  kill "$OLD_PID" 2>/dev/null && echo "Stopped old dashboard (pid $OLD_PID)" || true
  rm -f "$PID_FILE"
fi

nohup python3 -m uvicorn src.dashboard.server:app --host 0.0.0.0 --port 8080 \
    >> "$LOG_FILE" 2>&1 &
echo $! > "$PID_FILE"
echo "Dashboard running at http://100.78.161.126:8080 (pid $(cat "$PID_FILE"))"
