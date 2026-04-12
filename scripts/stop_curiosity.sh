#!/bin/bash
# Stop the Curiosity master process (and all its daemon children)
PID_FILE="$HOME/curiosity/curiosity.pid"

PID=$(cat "$PID_FILE" 2>/dev/null)
if [ -n "$PID" ]; then
    if kill -0 "$PID" 2>/dev/null; then
        kill "$PID" && echo "Curiosity stopped (PID: $PID)." && rm -f "$PID_FILE"
    else
        echo "Process $PID is not running (stale PID file)."
        rm -f "$PID_FILE"
    fi
else
    echo "No PID file found — Curiosity may not be running."
fi
