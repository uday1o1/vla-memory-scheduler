"""The demotion table from the committed eviction runs, without a GPU.

    python experiments/placement/eviction_analyze.py
    python experiments/placement/eviction_analyze.py --suffix ""   # a fresh run's files

Reads the four runs of eviction.py, each mechanism in both orders, and prints
the median seconds of every kind of change, the steady call time at each
residency level, and the largest element-wise difference between outputs,
within each run across residency levels and between mechanisms.
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

DATA = Path(__file__).resolve().parents[2] / "data"


def load(directory: Path, mech: str, order: str, suffix: str) -> dict:
    tag = "_order2" if order == "2" else ""
    return json.loads((directory / f"eviction_{mech}_nested{tag}{suffix}.json").read_text())


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dir", type=Path, default=DATA)
    p.add_argument("--suffix", default="_3090", help="filename suffix, '' for a fresh run")
    args = p.parse_args()
    runs = {(m, o): load(args.dir, m, o, args.suffix) for m in ("copy", "clean") for o in ("1", "2")}
    print(f"{runs['copy', '1']['meta'].get('gpu', 'unknown GPU')}, "
          f"host copies held {runs['clean', '1']['host_copy_gb']:.2f}GB\n")

    def medians(r):
        kinds = {}
        for t in r["transitions"]:
            kinds.setdefault((t["promoted"], t["demoted"]), []).append(t["total_s"])
        return {k: statistics.median(v) for k, v in kinds.items()}

    med = {key: medians(r) for key, r in runs.items()}
    print(f"{'change':<14}{'copy 1':<9}{'copy 2':<9}{'clean 1':<9}{'clean 2':<9}{'ratio'}")
    for pr, de in sorted(med["copy", "1"], key=lambda x: (x[0] > 0, x[1] or x[0])):
        label = f"demote {de}" if de else f"promote {pr}"  # nested: never both
        cells = [med[m, o][(pr, de)] for m in ("copy", "clean") for o in ("1", "2")]
        ratio = statistics.mean(cells[:2]) / statistics.mean(cells[2:])
        print(f"{label:<14}" + "".join(f"{c:<9.2f}" for c in cells)
              + (f"{ratio:.1f}x" if de else ""))

    print("\nSteady call seconds after each change, median by residency level:")
    for key, r in runs.items():
        by_k = {}
        for t in r["transitions"]:
            by_k.setdefault(t["k"], []).append(t["call_s"])
        print(f"  {key[0]:<6}order {key[1]}  " + "  ".join(
            f"K={k} {statistics.median(v):.2f}" for k, v in sorted(by_k.items(), reverse=True)))

    def maxdiff(a, b):
        return max(abs(x - y) for x, y in zip(a, b))

    print("\nLargest element-wise output difference:")
    for key, r in runs.items():
        print(f"  {key[0]:<6}order {key[1]}  across residency levels "
              f"{r['max_abs_diff_across_k']:.3e}")
    for o in ("1", "2"):
        a, b = runs["copy", o]["trajectories"], runs["clean", o]["trajectories"]
        print(f"  copy against clean, order {o}  " + ", ".join(
            f"K={k} {maxdiff(a[k], b[k]):.3e}" for k in sorted(set(a) & set(b), key=int)))


if __name__ == "__main__":
    main()
