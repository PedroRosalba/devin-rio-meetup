# CUDA-Oxide setup (this repo)

Follows [NVlabs installation guide](https://nvlabs.github.io/cuda-oxide/getting-started/installation.html).

**You cannot run `cargo oxide run vecadd` on macOS** — need Linux, Ampere+ GPU, driver **580+**, CUDA toolkit **13+**.

## Fastest path: RunPod + SSH

### 1. One-time on your Mac

```bash
# SSH key (skip if you already have ~/.ssh/id_ed25519.pub)
ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519 -N '' -C runpod-devin-meetup

# Paste into RunPod → Settings → SSH Public Keys:
cat ~/.ssh/id_ed25519.pub

# RunPod API key → Settings → API Keys
export RUNPOD_API_KEY='rp_...'
```

### 2. Create GPU pod + get SSH command

```bash
cd /path/to/devin-meetup-rio
bash scripts/runpod_provision_gpu.sh
```

The script prints the **SSH command** (proxy via `ssh.runpod.io`). Save it.

Default GPU: **RTX A5000**, image: **`nvidia/cuda:13.0.0-devel-ubuntu24.04`**.

### 3. Install CUDA-Oxide on the pod

```bash
export RUNPOD_SSH='ssh <pod-user>@ssh.runpod.io -i ~/.ssh/id_ed25519'
bash scripts/runpod_remote_bootstrap.sh
```

Or SSH in manually and run:

```bash
bash scripts/cuda_oxide_host_setup.sh
```

Success looks like:

```text
cargo oxide doctor   # all checks green / GPU seen
cargo oxide run vecadd
```

## Dev container (on the GPU host)

If the pod has Docker + [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html):

```bash
npx -y @devcontainers/cli up --workspace-folder .
npx -y @devcontainers/cli exec --workspace-folder . cargo oxide doctor
npx -y @devcontainers/cli exec --workspace-folder . cargo oxide run vecadd
```

Uses `.devcontainer/` (same stack as upstream cuda-oxide).

## After vecadd works

- Keep `third_party/cuda-oxide` as the upstream reference; build meetup kernels under `cuda-oxide/` in this repo.
- CPU oracle on the pod: `./scripts/setup_env.sh` then `python tools/verify_hf.py --strict`.
