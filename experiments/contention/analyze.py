"""Analysis of the compute-contention experiment (the negative boundary result).

Tests whether residency adaptation changes deadline-miss rate under compute
contention. The hypothesis under test is that it does NOT, because compute
interference is proportional across residency levels - so residency has no
lever against it.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from scipy.stats import wilcoxon

SCENARIOS = ["no_contention", "steady", "bursty"]


def miss_rate(latencies, deadline):
    return sum(1 for l in latencies if l > deadline) / len(latencies)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path,
                   default=Path("data/compute_contention_3090.json"))
    args = p.parse_args()

    data = json.loads(args.input.read_text())
    print(f"Reps available: {len(data)}\n")

    rows = {s: {"baseline": [], "adaptive": []} for s in SCENARIOS}
    for clip, rec in data.items():
        deadline = rec["deadline"]
        print(f"clip {clip[:8]}  deadline={deadline:.3f}s")
        for s in SCENARIOS:
            bkey = "no_contention_baseline" if s == "no_contention" else f"{s}_baseline"
            akey = "no_contention_adaptive" if s == "no_contention" else f"{s}_adaptive"
            if bkey not in rec or akey not in rec:
                continue
            b = rec[bkey]; a = rec[akey]
            mb = miss_rate(b["latencies"], deadline)
            ma = miss_rate(a["latencies"], deadline)
            meanb = sum(b["latencies"]) / len(b["latencies"])
            meana = sum(a["latencies"]) / len(a["latencies"])
            ks_used = sorted(set(a["residency_k"]))
            transitions = sum(1 for t in a["transition_s"] if t > 0)
            rows[s]["baseline"].append(mb)
            rows[s]["adaptive"].append(ma)
            print(f"  {s:<14} miss: base={mb:.2f} adapt={ma:.2f}   "
                  f"mean: base={meanb:.2f}s adapt={meana:.2f}s   "
                  f"adaptive K used={ks_used} transitions={transitions}")
        print()

    print("=== Paired comparison across reps (miss rate) ===")
    for s in SCENARIOS:
        b, a = rows[s]["baseline"], rows[s]["adaptive"]
        if len(b) < 2:
            print(f"{s:<14} n={len(b)} - INSUFFICIENT_EVIDENCE for a paired test")
            continue
        diffs = [ai - bi for ai, bi in zip(a, b)]
        if all(d == 0 for d in diffs):
            print(f"{s:<14} n={len(b)}  all paired differences are exactly zero "
                  f"-> no test possible; arms are identical")
            continue
        try:
            stat, pval = wilcoxon(a, b)
            print(f"{s:<14} n={len(b)}  W={stat:.1f} p={pval:.4f}  "
                  f"mean diff={sum(diffs)/len(diffs):+.3f}")
        except Exception as e:
            print(f"{s:<14} n={len(b)}  test failed: {e}")

    print("\nNote: with n=2 reps this is descriptive, not inferential. The locked")
    print("design calls for n=6; these two reps were collected before the design")
    print("flaw was found and are retained only as the compute-contention boundary")
    print("evidence, where the arms are expected to be identical by construction.")


if __name__ == "__main__":
    main()
