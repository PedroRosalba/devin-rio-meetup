#!/usr/bin/env bash
# Create a RunPod GPU instance with SSH and run cuda_oxide_host_setup.sh remotely.
# Requires: RUNPOD_API_KEY, SSH public key registered at https://www.runpod.io/console/user/settings
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
API_KEY="${RUNPOD_API_KEY:-}"
GPU_TYPE="${RUNPOD_GPU_TYPE:-NVIDIA RTX A5000}"
IMAGE="${RUNPOD_IMAGE:-nvidia/cuda:13.0.0-devel-ubuntu24.04}"
POD_NAME="${RUNPOD_POD_NAME:-devin-meetup-cuda-oxide}"

if [[ -z "$API_KEY" ]]; then
  echo "Set RUNPOD_API_KEY (RunPod console → Settings → API Keys)."
  echo "Also register your SSH public key, then re-run this script."
  exit 1
fi

KEY_FILE="${RUNPOD_SSH_KEY:-$HOME/.ssh/id_ed25519}"
PUB_FILE="${KEY_FILE}.pub"
if [[ ! -f "$PUB_FILE" ]]; then
  echo "No $PUB_FILE — run: ssh-keygen -t ed25519 -f $KEY_FILE -N '' -C runpod-devin-meetup"
  exit 1
fi
PUBKEY="$(tr -d '\n' < "$PUB_FILE")"

echo "==> Creating RunPod (GPU: $GPU_TYPE)"
CREATE_JSON="$(curl -sf -H "Authorization: Bearer ${API_KEY}" -H "Content-Type: application/json" \
  -d "$(jq -n \
    --arg name "$POD_NAME" \
    --arg image "$IMAGE" \
    --arg gpu "$GPU_TYPE" \
    --arg pubkey "$PUBKEY" \
    '{
      name: $name,
      image: $image,
      gpu: { count: 1, type: $gpu },
      ports: ["22/tcp"],
      startSsh: true,
      env: { PUBLIC_KEY: $pubkey },
      disk: { container: 50 }
    }')" \
  "https://rest.runpod.io/v2/pods" 2>/dev/null || true)"

if [[ -z "$CREATE_JSON" ]] || ! echo "$CREATE_JSON" | jq -e '.id' >/dev/null 2>&1; then
  echo "REST v2 create failed; trying GraphQL..."
  CREATE_JSON="$(curl -sf -H "Content-Type: application/json" \
    --url "https://api.runpod.io/graphql?api_key=${API_KEY}" \
    --data "$(jq -n \
      --arg name "$POD_NAME" \
      --arg image "$IMAGE" \
      --arg gpu "$GPU_TYPE" \
      --arg pubkey "$PUBKEY" \
      '{
        query: "mutation($input: PodFindAndDeployOnDemandInput!) { podFindAndDeployOnDemand(input: $input) { id desiredStatus runtime { ports { ip isIpPublic privatePort publicPort } } } }",
        variables: {
          input: {
            cloudType: "ALL",
            gpuCount: 1,
            volumeInGb: 50,
            containerDiskInGb: 50,
            minVcpuCount: 4,
            minMemoryInGb: 30,
            gpuTypeId: $gpu,
            name: $name,
            imageName: $image,
            dockerArgs: "",
            ports: "22/tcp",
            volumeMountPath: "/workspace",
            env: [{ key: "PUBLIC_KEY", value: $pubkey }]
          }
        }
      }')" )"
  POD_ID="$(echo "$CREATE_JSON" | jq -r '.data.podFindAndDeployOnDemand.id // empty')"
else
  POD_ID="$(echo "$CREATE_JSON" | jq -r '.id')"
fi

if [[ -z "${POD_ID:-}" ]]; then
  echo "Failed to create pod:"
  echo "$CREATE_JSON" | jq . 2>/dev/null || echo "$CREATE_JSON"
  exit 1
fi

echo "Pod ID: $POD_ID"
echo "Waiting for SSH (up to ~10 min)..."
SSH_CMD=""
for _ in $(seq 1 60); do
  STATUS_JSON="$(curl -sf -H "Authorization: Bearer ${API_KEY}" \
    "https://rest.runpod.io/v2/pods/${POD_ID}" 2>/dev/null || true)"
  if [[ -n "$STATUS_JSON" ]]; then
    SSH_CMD="$(echo "$STATUS_JSON" | jq -r '.ssh.proxy.command // .ssh.direct.command // empty')"
    STATE="$(echo "$STATUS_JSON" | jq -r '.status // .desiredStatus // empty')"
    [[ -n "$SSH_CMD" && "$SSH_CMD" != "null" ]] && break
    echo "  status=$STATE (waiting for ssh block...)"
  else
    GQL="$(curl -sf -H "Content-Type: application/json" \
