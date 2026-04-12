#!/bin/bash
# Show running status of all Curiosity daemons and recent log output

PID_FILE="$HOME/curiosity/curiosity.pid"
LOG_DIR="$HOME/curiosity/logs"

echo "════════════════════════════════════════════════"
echo "  Curiosity System Status"
echo "════════════════════════════════════════════════"

# Master process
MASTER_PID=$(cat "$PID_FILE" 2>/dev/null)
if [ -n "$MASTER_PID" ] && kill -0 "$MASTER_PID" 2>/dev/null; then
    echo "  Master:     RUNNING  (PID: $MASTER_PID)"
else
    echo "  Master:     STOPPED"
fi

echo ""
echo "  Daemon processes (children of master):"
DAEMONS=("assessor" "formulator" "solver" "verifier" "memorizer")
for DAEMON in "${DAEMONS[@]}"; do
    COUNT=$(pgrep -fc "src.${DAEMON}.${DAEMON}" 2>/dev/null || echo 0)
    if [ "$COUNT" -gt 0 ]; then
        PID=$(pgrep -f "src.${DAEMON}.${DAEMON}" | head -1)
        echo "  ${DAEMON}:  RUNNING  (PID: $PID)"
    else
        echo "  ${DAEMON}:  STOPPED"
    fi
done

echo ""
echo "════════════════════════════════════════════════"
echo "  Recent master log (last 20 lines):"
echo "────────────────────────────────────────────────"
if [ -f "$LOG_DIR/curiosity_master.log" ]; then
    tail -n 20 "$LOG_DIR/curiosity_master.log"
else
    echo "  (no log file found at $LOG_DIR/curiosity_master.log)"
fi
echo "════════════════════════════════════════════════"
