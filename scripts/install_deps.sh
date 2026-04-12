#!/bin/bash
# Install all Python deps for Curiosity on DGX
set -euo pipefail

echo "Installing Python dependencies for Curiosity …"
pip install redis chromadb requests
echo "Dependencies installed."
