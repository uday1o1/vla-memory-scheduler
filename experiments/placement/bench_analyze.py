"""Transition speedup and latency parity from the committed placement runs,
without a GPU.

    python experiments/placement/bench_analyze.py
    python experiments/placement/bench_analyze.py --dir results   # a fresh run

For every run of benchmark.py: the mean transition time of each placement at
each adaptation granularity, its speedup and the layers each moved; and the
steady-state latency difference, nested against upstream, at each residency
level. Runs at --max-k 14 follow the same adaptation paths on every machine,
so their move counts match and their speedups can be compared. Latency on
one RTX 3090 depends on which placement is measured first, so where a run in
the other order exists the two are also averaged.
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

DATA = Path(__file__).resolve().parents[2] / "data"
# Runs of the same paths in the opposite order: (first measured upstream, nested).
BOTH_ORDERS = [("placement_bench_3090new_k14.json", "placement_bench_3090new_swapped.json"),
               ("placement_bench_4090_k14.json", "placement_bench_4090_k14_nestedfirst.json"),
               ("placement_bench.json", "placement_bench_nestedfirst.json")]  # reproduce.sh


def pct(n: float, u: float) -> float:
    return 100 * (n - u) / u


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dir", type=Path, default=DATA)
    args = p.parse_args()
    runs = {f.name: json.loads(f.read_text()) for f in sorted(args.dir.glob("placement_bench*.json"))}
    print(f"{'run':<36}{'|dK|':>5}  " + "".join(f"{g:>12}" for g in ("1", "2", "3", "5", "8", "12")))
    for name, d in runs.items():
        t = d["transition"]
        print(f"{name[15:-5].lstrip('_') or 'this run':<36}{'x':>5}  "
              + "".join(f"{t[g]['upstream']['mean'] / t[g]['nested']['mean']:>11.2f}x" for g in t))
        print(f"{'':<36}{'moves':>5}  "
              + "".join(f"{str(t[g]['upstream']['moves']) + '/' + str(t[g]['nested']['moves']):>12}" for g in t))

    print("\nSteady-state latency, nested against upstream, mean over clips:")
    for name, d in runs.items():
        lat = d["latency"]
        per_k = {k: statistics.mean(pct(c["nested"], c["upstream"]) for c in v.values())
                 for k, v in lat.items()}
        every = [pct(c["nested"], c["upstream"]) for v in lat.values() for c in v.values()]
        print(f"  {name[15:-5].lstrip('_') or 'this run':<34}" + "  ".join(f"K={k} {p:+.3f}%" for k, p in per_k.items())
              + f"   all {statistics.mean(every):+.3f}%")
    print("\nAveraged over both measurement orders:")
    for a_name, b_name in BOTH_ORDERS:
        if a_name not in runs or b_name not in runs:
            continue
        a, b = runs[a_name]["latency"], runs[b_name]["latency"]
        both = [pct(a[k][c]["nested"] + b[k][c]["nested"], a[k][c]["upstream"] + b[k][c]["upstream"])
                for k in a for c in a[k]]
        print(f"  {a_name[15:-5].lstrip('_') or 'this run'} with {b_name[15:-5].lstrip('_') or 'this run'}: {statistics.mean(both):+.3f}% "
              f"over {len(both)} clip and level pairs")


if __name__ == "__main__":
    main()
