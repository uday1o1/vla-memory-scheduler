#!/bin/bash
# Prepare a fresh rented GPU instance to run these experiments.
#
# The upstream README is not sufficient on its own. Three things it does not
# say:
#   1. oom-free-alpamayo needs the alpamayo_r1 package, which lives in a
#      separate repository (NVlabs/alpamayo) that the README does not name.
#   2. Plain pip fails on flash-attn, which imports torch during its own build.
#      uv with --no-build-isolation-package is what works.
#   3. accelerate is required but listed nowhere.
#
# The environment needs Python 3.12: alpamayo_r1 declares requires-python
# ==3.12.*, so installing it fails on other versions.
#
# Installing alpamayo_r1 downgrades torch to its pinned 2.8.0. CUDA still works
# after the downgrade; this is expected, not a fault.
#
# The dataset is gated: request access on its Hugging Face page, then use any
# read token. A fine-grained token additionally needs read access to public
# gated repositories. The model itself is not gated.
#
# Usage:  bash setup/setup_instance.sh
#   WORKDIR   where the upstream clones live (default: parent of this repo)
#   VENV      a virtualenv to activate, if not already active
#   HF_TOKEN  overrides the stored Hugging Face token
#   OOM_FREE_ALPAMAYO_COMMIT, ALPAMAYO_COMMIT
#             upstream commits to check out; the defaults are the commits
#             recorded in data/compute_interaction_3090.json

set -euo pipefail

# Fall back to the token huggingface_hub already stores, so a machine that has
# been authenticated once does not need the variable exported again.
if [ -z "${HF_TOKEN:-}" ] && [ -s "${HF_HOME:-$HOME/.cache/huggingface}/token" ]; then
    HF_TOKEN="$(cat "${HF_HOME:-$HOME/.cache/huggingface}/token")"
fi

if [ -z "${HF_TOKEN:-}" ]; then
    echo "No Hugging Face token found."
    echo "The PhysicalAI-Autonomous-Vehicles dataset is gated. Request access on its"
    echo "Hugging Face page, create a read token at huggingface.co/settings/tokens"
    echo "(a fine-grained token needs read access to public gated repositories),"
    echo "then either run:  HF_TOKEN=hf_... bash setup/setup_instance.sh"
    echo "or place it at:   \${HF_HOME:-\$HOME/.cache/huggingface}/token"
    exit 1
fi

# WORKDIR is where the upstream clones go. It defaults to the
# parent of this repository, so scheduler/paths.py finds the clones as siblings
# without configuration. Override it for a machine laid out differently.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKDIR="${WORKDIR:-$(dirname "$REPO_ROOT")}"
mkdir -p "$WORKDIR"

echo "=== System packages ==="
# DCGM is not installed: none of the experiments here use it, and its
# profiling counters are commonly unavailable inside shared cloud containers.
if command -v apt-get >/dev/null 2>&1 && [ "$(id -u)" = "0" ]; then
    apt-get update -qq
    apt-get install -y -qq git wget python3-pip
else
    echo "  skipped: needs apt-get and root. Ensure git, wget and pip are present."
fi

# Some images ship CUDA forward-compatibility libraries and register them ahead
# of the host driver. Forward compatibility is unsupported on GeForce cards, so
# CUDA then fails to initialise with error 804 even though nvidia-smi works.
# Unregister them on GeForce cards.
if [ "$(id -u)" = "0" ] && ls /etc/ld.so.conf.d/*compat* >/dev/null 2>&1 \
   && nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | grep -q GeForce; then
    for f in /etc/ld.so.conf.d/*compat*; do
        grep -q "cuda.*compat" "$f" && mv "$f" "$f.disabled"
    done
    ldconfig
    echo "  CUDA forward-compatibility libraries unregistered (unsupported on this GPU)"
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
OOM_FREE_ALPAMAYO_COMMIT="${OOM_FREE_ALPAMAYO_COMMIT:-b724a081af5a976c7a412333763ad142150f3dda}"
ALPAMAYO_COMMIT="${ALPAMAYO_COMMIT:-11a0e01c13a5622377c45ee37d653351453ec43b}"
cd "$WORKDIR"
[ -d oom-free-alpamayo ] || git clone -q https://github.com/aveeslab/oom-free-alpamayo.git
[ -d alpamayo ] || git clone -q https://github.com/NVlabs/alpamayo.git
git -C oom-free-alpamayo checkout -q "$OOM_FREE_ALPAMAYO_COMMIT"
git -C alpamayo checkout -q "$ALPAMAYO_COMMIT"
echo "  oom-free-alpamayo $(git -C oom-free-alpamayo rev-parse --short HEAD), alpamayo $(git -C alpamayo rev-parse --short HEAD)"

echo "=== Python packages ==="
# flash-attn imports torch during its own build, so it must not be built in
# isolation. uv exempts only flash-attn; without uv, pip disables isolation for
# every package, which needs torch and hatchling already installed.
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

echo "=== Model weights (about 22GB, one time) ==="
# Into the default Hugging Face cache, which is where the experiments' model
# loads look (set HF_HOME to move it).
python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download("nvidia/Alpamayo-R1-10B")
print("  model downloaded")
PY

echo "=== Profiling for this machine ==="
cd "$WORKDIR/oom-free-alpamayo"
# Clock locking needs privileges a container usually lacks, so --no-lock-clock
# is passed.
python scripts/profile.py --model r1 --no-lock-clock --output r1_config.json

echo
echo "Ready. The residency profiles in scheduler/profiles.py were measured on an"
echo "RTX 3090; on different hardware, measure them on the machine in use:"
echo "    python experiments/characterize/calibrate.py    # latency per residency level"
echo "    python experiments/characterize/vram_per_k.py   # footprint per residency level"
echo "    python experiments/characterize/build_profiles.py"
echo "  then run the policy with VLA_PROFILES pointing at the file it writes."
echo
echo "The placement results are hardware independent in move count and should"
echo "reproduce exactly; only the seconds per move should differ:"
echo "    python experiments/placement/theory.py      # move-optimality, no GPU needed"
echo "    python experiments/placement/benchmark.py   # transition cost and latency parity"
