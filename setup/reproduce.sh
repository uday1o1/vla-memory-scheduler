#!/bin/bash
# Re-run the measurements behind the committed results, in the order and with
# the arguments that produced them.
#
# Usage:  bash setup/reproduce.sh [core|placement|all]     (default: core)
#   core       characterization, demotion cost, and the trace experiment:
#              the primary results. About 24 hours on an RTX 3090.
#   placement  transition cost, latency parity, output equivalence and the
#              end-to-end placement comparison. Several hours.
#   VLA_RESULTS  where results go (default: results/ in this repository, so the
#                committed data/ is never overwritten)
#
# Each stage writes a .done marker, so running the script again after an
# interruption resumes at the first unfinished stage, which starts over. A failing stage writes FAILED with
# its name and stops. Output of every stage goes to $VLA_RESULTS/reproduce.log.
#
# Requirements: a 24GB NVIDIA GPU for the trace experiment (it checks before
# starting and refuses a card on which its design cannot separate the
# policies), at least 64GB of host memory since keeping a host copy pins the
# whole model, disk for the 21GB of model weights and the Python
# environment, and a Hugging Face token as described in
# setup_instance.sh.
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
WHAT="${1:-core}"
case "$WHAT" in core|placement|all) ;; *) echo "usage: $0 [core|placement|all]"; exit 2;; esac
[ -z "${VIRTUAL_ENV:-}" ] && [ -f /venv/main/bin/activate ] && source /venv/main/bin/activate
export VLA_RESULTS="${VLA_RESULTS:-$REPO/results}"
R="$VLA_RESULTS"; mkdir -p "$R"; LOG="$R/reproduce.log"

stage() {  # name, command...
  local name=$1; shift
  [ -f "$R/$name.done" ] && { echo "skip $name (done)"; return 0; }
  echo "=== $name $(date -u +%FT%TZ)" | tee -a "$LOG"
  if "$@" >>"$LOG" 2>&1; then touch "$R/$name.done"
  else echo "$name" >"$R/FAILED"; echo "FAILED $name, see $LOG" | tee -a "$LOG"; exit 1; fi
}

stage setup bash setup/setup_instance.sh

if [ "$WHAT" != placement ]; then
  # Latency and footprint across the residency range, and the three profiles.
  stage calibrate    python experiments/characterize/calibrate.py --k-values 0 8 12 16 20 24 28 30 33 --reps 5
  stage vram         python experiments/characterize/vram_per_k.py --k-values 33 28 24 20 16 12 8
  stage profiles     python experiments/characterize/build_profiles.py --k 33 24 16
  # Demotion with and without a kept host copy, each mechanism in both orders.
  stage evict_copy1  python experiments/placement/eviction.py --demotion copy  --output "$R/eviction_copy_nested.json"
  stage evict_clean1 python experiments/placement/eviction.py --demotion clean --output "$R/eviction_clean_nested.json"
  stage evict_clean2 python experiments/placement/eviction.py --demotion clean --output "$R/eviction_clean_nested_order2.json"
  stage evict_copy2  python experiments/placement/eviction.py --demotion copy  --output "$R/eviction_copy_nested_order2.json"
  # Static residency against switching under a changing tenant: one pilot
  # repetition, then eight. Runs that end in an error are repeated in place.
  export VLA_PROFILES="$R/profiles.json"
  stage trace_pilot  python experiments/trace/run.py --reps 1 --phase-calls 2 --output "$R/trace_pilot.json" --outdir "$R/trace_pilot"
  stage trace        python experiments/trace/run.py --reps 8 --phase-calls 6 2
  stage trace_rerun  python experiments/trace/run.py --rerun-failed
  # Compute contention, last, since it runs under CUDA MPS and nothing else does.
  nvidia-cuda-mps-control -d 2>/dev/null || echo "CUDA MPS could not be started; recorded in the output"
  stage contention   python experiments/contention/interaction.py --reps 8
  echo quit | nvidia-cuda-mps-control 2>/dev/null || true
fi

if [ "$WHAT" != core ]; then
  stage bench        python experiments/placement/benchmark.py --max-k 14 --transition-reps 5 --latency-reps 5 --latency-warmup 3
  stage bench_swap   python experiments/placement/benchmark.py --max-k 14 --transition-reps 1 --latency-reps 5 --latency-warmup 3 --nested-first --output "$R/placement_bench_nestedfirst.json"
  stage correctness  python experiments/placement/correctness.py
  stage adaptive     python experiments/placement/adaptive_cost.py --k-a 24 --k-b 16 --phases 6 --calls-per-phase 4
  stage spread       python experiments/placement/spread_latency.py --k-values 16 24 33 --reps 5 --warmup 3
fi

touch "$R/ALLDONE"
echo
echo "Done. Compare against the committed results with:"
[ "$WHAT" != placement ] && cat <<MSG
  python experiments/placement/eviction_analyze.py --dir "$R" --suffix ""
  python experiments/trace/analyze.py --input "$R/trace_results.json"
  python experiments/trace/plot.py --input "$R/trace_results.json" --outdir "$R/figures"
MSG
[ "$WHAT" != core ] && echo "  python experiments/placement/bench_analyze.py --dir \"$R\""
echo "  and data/calibration_3090.json, data/vram_per_k_3090.json against $R/."
