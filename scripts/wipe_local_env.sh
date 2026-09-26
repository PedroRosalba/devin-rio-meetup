#!/usr/bin/env bash
# Remove local demo artifacts under this repo (safe to run before deleting the whole folder).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "Removing $ROOT/.venv"
rm -rf .venv

echo "Removing $ROOT/models"
rm -rf models

echo "Removing Python caches"
find . -type d -name __pycache__ -prune -exec rm -rf {} + 2>/dev/null || true

echo "Done. Re-run ./scripts/setup_env.sh when you need the CPU demo again."
echo "Modal token (if any): rm -f ~/.modal.toml"
