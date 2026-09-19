#!/bin/bash
# Prepare a fresh rented GPU instance to run these experiments.
#
# The upstream README is not sufficient on its own. Three things it does not
# say, each of which cost setup time to discover:
#   1. oom-free-alpamayo needs the alpamayo_r1 package, which lives in a
#      separate repository (NVlabs/alpamayo) that the README does not name.
#   2. Plain pip fails on flash-attn, which imports torch during its own build.
#      uv with --no-build-isolation-package is what works.
#   3. accelerate is required but listed nowhere.
#
# Installing alpamayo_r1 downgrades torch to its pinned 2.8.0. CUDA still works
# after the downgrade; this is expected, not a fault.
#
# The dataset is gated and needs a token whose "read public gated repos"
# permission is explicitly enabled. That permission is not granted by default,
# including for fine-grained tokens, and access fails with 403 without it.
#
# Usage:  HF_TOKEN=hf_... bash setup_instance.sh

set -euo pipefail

# Fall back to the token huggingface_hub already stores, so a machine that has
# been authenticated once does not need the variable exported again.
if [ -z "${HF_TOKEN:-}" ] && [ -s "${HF_HOME:-$HOME/.cache/huggingface}/token" ]; then
    HF_TOKEN="$(cat "${HF_HOME:-$HOME/.cache/huggingface}/token")"
fi

if [ -z "${HF_TOKEN:-}" ]; then
    echo "No Hugging Face token found."
    echo "The PhysicalAI-Autonomous-Vehicles dataset is gated. Create a token at"
    echo "huggingface.co/settings/tokens with 'read public gated repos' enabled,"
    echo "then either run:  HF_TOKEN=hf_... bash setup_instance.sh"
    echo "or place it at:   \${HF_HOME:-\$HOME/.cache/huggingface}/token"
    exit 1
fi

echo "=== System packages ==="
# DCGM is deliberately not installed. Its profiling module fails to load in
# these containers, so the fine-grained occupancy counters it exists for are
# unavailable anyway, and none of the experiments here use it. It is a 911MB
# download that stalled setup for nine minutes on one host before being cut.
apt-get update -qq
apt-get install -y -qq git wget python3-pip

source /venv/main/bin/activate

echo "=== Repositories ==="
cd /root
[ -d oom-free-alpamayo ] || git clone -q https://github.com/aveeslab/oom-free-alpamayo.git
[ -d alpamayo ] || git clone -q https://github.com/NVlabs/alpamayo.git

echo "=== Python packages ==="
cd /root/alpamayo
uv pip install -q -e . --no-build-isolation-package flash-attn
cd /root/oom-free-alpamayo
pip install -q -e .
uv pip install -q accelerate

echo "=== Verifying CUDA ==="
python - <<'PY'
import torch
print(f"  torch {torch.__version__}  cuda {torch.version.cuda}  available={torch.cuda.is_available()}")
p = torch.cuda.get_device_properties(0)
print(f"  {p.name}  {p.total_memory / 1024**3:.2f}GB  cc {p.major}.{p.minor}")
PY

echo "=== Hugging Face authentication ==="
python - <<PY
from huggingface_hub import login, HfApi
login(token="${HF_TOKEN}", add_to_git_credential=False)
api = HfApi()
api.dataset_info("nvidia/PhysicalAI-Autonomous-Vehicles")
print("  dataset reachable")
PY

echo "=== Model weights (about 21GB, one time) ==="
python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download("nvidia/Alpamayo-R1-10B", cache_dir="/root/hf_cache")
print("  model downloaded")
PY

echo "=== Profiling for this machine ==="
cd /root/oom-free-alpamayo
# Clock locking is denied inside these containers even as root, so timing runs
# cannot pin clocks. --no-lock-clock is required, not optional.
python scripts/profile.py --model r1 --no-lock-clock --output r1_config.json

echo
echo "Ready. The residency profiles in scheduler/profiles.py were measured on an"
echo "RTX 3090 and must be re-measured on different hardware before use:"
echo "    python experiments/characterize/calibrate.py    # latency per residency level"
echo "    python experiments/characterize/vram_per_k.py   # footprint per residency level"
echo
echo "The placement results are hardware independent in move count and should"
echo "reproduce exactly; only the seconds per move should differ:"
echo "    python experiments/placement/theory.py      # move-optimality, no GPU needed"
echo "    python experiments/placement/benchmark.py   # transition cost and latency parity"
