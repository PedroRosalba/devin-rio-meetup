#!/usr/bin/env bash
# Push bootstrap to an already-running RunPod over SSH.
# Example: RUNPOD_SSH='ssh abc123-6080@ssh.runpod.io -i ~/.ssh/id_ed25519' bash scripts/runpod_remote_bootstrap.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
POD_ID="${RUNPOD_POD_ID:-}"
SSH_CMD="${RUNPOD_SSH:-}"
KEY_FILE="${RUNPOD_SSH_KEY:-$HOME/.ssh/id_ed25519}"

if [[ -z "$SSH_CMD" && -n "$POD_ID" && -n "${RUNPOD_API_KEY:-}" ]]; then
  command -v jq >/dev/null
  STATUS_JSON="$(curl -sf -H "Authorization: Bearer ${RUNPOD_API_KEY}" \
    "https://rest.runpod.io/v2/pods/${POD_ID}")"
  SSH_CMD="$(echo "$STATUS_JSON" | jq -r '.ssh.proxy.command // empty')"
fi

if [[ -z "$SSH_CMD" ]]; then
  echo "Set RUNPOD_SSH to the ssh command from RunPod console, or RUNPOD_POD_ID + RUNPOD_API_KEY."
  exit 1
fi

REMOTE_DIR="/workspace/devin-meetup-rio"
echo "==> Upload repo tarball"
tar -C "$ROOT" -czf - --exclude .venv --exclude models --exclude third_party/cuda-oxide . | \
  eval "$SSH_CMD" "mkdir -p $REMOTE_DIR && tar -xzf - -C $REMOTE_DIR"

echo "==> Remote CUDA-Oxide setup"
eval "$SSH_CMD" "cd $REMOTE_DIR && bash scripts/cuda_oxide_host_setup.sh"
