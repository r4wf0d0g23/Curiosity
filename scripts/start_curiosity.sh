#!/bin/bash
# Start the full Curiosity system
set -euo pipefail

LOG_DIR="$HOME/curiosity/logs"
PID_FILE="$HOME/curiosity/curiosity.pid"
CODE_DIR="/home/rawdata/curiosity_code"

mkdir -p "$LOG_DIR"

if [ -f "$PID_FILE" ]; then
    OLD_PID=$(cat "$PID_FILE")
    if kill -0 "$OLD_PID" 2>/dev/null; then
        echo "Curiosity is already running (PID: $OLD_PID). Use stop_curiosity.sh first."
        exit 1
    else
        echo "Stale PID file found — cleaning up."
        rm -f "$PID_FILE"
    fi
fi

cd "$CODE_DIR"
nohup python3 src/curiosity.py >> "$LOG_DIR/master.log" 2>&1 &
echo $! > "$PID_FILE"
echo "Curiosity started (PID: $(cat "$PID_FILE"))"
