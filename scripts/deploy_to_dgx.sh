#!/bin/bash
# Pull latest code to DGX and restart Curiosity
set -euo pipefail

DGX_HOST="rawdata@100.78.161.126"

echo "Deploying to DGX ($DGX_HOST) …"

ssh "$DGX_HOST" "
  set -e
  cd ~/curiosity_code
  git pull
  bash scripts/stop_curiosity.sh 2>/dev/null || true
  sleep 2
  bash scripts/start_curiosity.sh
"

echo "Deployed and restarted."
