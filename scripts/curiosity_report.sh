#!/bin/bash
# Print current Curiosity capability status report.
# Usage: ./scripts/curiosity_report.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

exec python3 "$REPO_ROOT/src/metrics/status.py" "$@"
