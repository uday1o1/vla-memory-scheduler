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
# Usage:  bash setup/setup_instance.sh
#   WORKDIR   where the upstream clones live (default: parent of this repo)
#   VENV      a virtualenv to activate, if not already active
#   HF_TOKEN  overrides the stored Hugging Face token

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

# WORKDIR is where the upstream clones and model cache go. It defaults to the
# parent of this repository, so scheduler/paths.py finds the clones as siblings
# without configuration. Override it for a machine laid out differently.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKDIR="${WORKDIR:-$(dirname "$REPO_ROOT")}"
mkdir -p "$WORKDIR"

echo "=== System packages ==="
# DCGM is deliberately not installed. Its profiling module fails to load in
# these containers, so the fine-grained occupancy counters it exists for are
# unavailable anyway, and none of the experiments here use it. It is a 911MB
# download that stalled setup for nine minutes on one host before being cut.
if command -v apt-get >/dev/null 2>&1 && [ "$(id -u)" = "0" ]; then
    apt-get update -qq
    apt-get install -y -qq git wget python3-pip
else
    echo "  skipped: needs apt-get and root. Ensure git, wget and pip are present."
fi

# Activate a virtualenv only if one is present and none is already active.
# Rented images often ship one; a local machine usually does not.
if [ -z "${VIRTUAL_ENV:-}" ] && [ -n "${VENV:-}" ] && [ -f "$VENV/bin/activate" ]; then
    source "$VENV/bin/activate"
elif [ -z "${VIRTUAL_ENV:-}" ] && [ -f /venv/main/bin/activate ]; then
    source /venv/main/bin/activate
fi
PIP="pip"
command -v uv >/dev/null 2>&1 && PIP="uv pip"

echo "=== Repositories ==="
cd "$WORKDIR"
[ -d oom-free-alpamayo ] || git clone -q https://github.com/aveeslab/oom-free-alpamayo.git
[ -d alpamayo ] || git clone -q https://github.com/NVlabs/alpamayo.git

echo "=== Python packages ==="
# flash-attn imports torch during its own build, so it must not be built in
# isolation. Plain pip fails here; uv with the exemption is what works.
cd "$WORKDIR/alpamayo"
if [ "$PIP" = "uv pip" ]; then
    uv pip install -q -e . --no-build-isolation-package flash-attn
else
    pip install -q -e . --no-build-isolation
fi
cd "$WORKDIR/oom-free-alpamayo"
pip install -q -e .
$PIP install -q accelerate scipy

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
import os
from huggingface_hub import snapshot_download
snapshot_download("nvidia/Alpamayo-R1-10B", cache_dir=os.environ.get("MODEL_CACHE") or os.path.expanduser("~/hf_cache"))
print("  model downloaded")
PY

echo "=== Profiling for this machine ==="
cd "$WORKDIR/oom-free-alpamayo"
# Clock locking is denied inside these containers even as root, so timing runs
# cannot pin clocks. --no-lock-clock is required, not optional.
python scripts/profile.py --model r1 --no-lock-clock --output r1_config.json

echo
echo "Ready. The residency profiles in scheduler/profiles.py were measured on an"
echo "RTX 3090 and must be re-measured on different hardware before use:"
echo "    python experiments/characterize/calibrate.py    # latency per residency level"
echo "    python experiments/characterize/vram_per_k.py   # footprint per residency level"
echo "    python experiments/characterize/build_profiles.py"
echo "  then run the policy with VLA_PROFILES pointing at the file it writes."
echo
echo "The placement results are hardware independent in move count and should"
echo "reproduce exactly; only the seconds per move should differ:"
echo "    python experiments/placement/theory.py      # move-optimality, no GPU needed"
echo "    python experiments/placement/benchmark.py   # transition cost and latency parity"
