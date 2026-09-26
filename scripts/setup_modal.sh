#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ ! -d .venv ]]; then
  echo "Run ./scripts/setup_env.sh first (creates .venv)."
  exit 1
fi
# shellcheck disable=SC1091
source .venv/bin/activate
pip install -q -U pip
pip install -q -r modal/requirements.txt

echo "==> Modal CLI"
python -m modal --version

echo ""
echo "Next (interactive, opens browser):"
echo "  python3 -m modal setup"
echo ""
echo "Then:"
echo "  modal run modal/get_started.py"
