"""Analysis of the static versus profile-switching experiment.

Primary metric is deadline-miss rate, as the professor asked. Completion rate
is reported alongside it because the two arms can fail differently: a run that
never produces a trajectory misses every deadline, but it is a different
failure from one that produces trajectories too slowly, and collapsing them
would hide which is happening.

The paired Wilcoxon signed-rank test from the original proposal is applied per
pressure level across repetitions. Where every paired difference is zero the
test is not reported, since a test on identical arms carries no information;
that case is stated plainly instead.
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

from scipy.stats import wilcoxon


def rank_biserial(a: list[float], b: list[float]) -> float:
    """Matched-pairs rank-biserial correlation, the effect size the proposal named."""
    diffs = [x - y for x, y in zip(a, b) if x != y]
    if not diffs:
        return 0.0
    ranked = sorted(diffs, key=abs)
    pos = sum(i + 1 for i, d in enumerate(ranked) if d > 0)
    neg = sum(i + 1 for i, d in enumerate(ranked) if d < 0)
    total = pos + neg
    return (pos - neg) / total if total else 0.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, default=Path("data/switch_results.json"))
    args = p.parse_args()

    records = json.loads(args.input.read_text())
    if not records:
        print("no records")
        return

    deadline = records[0]["deadline_s"]
    print(f"Deadline: {deadline:.3f}s\n")

    by = defaultdict(dict)
    for r in records:
        by[r["pressure_gb"]][r["policy"]] = r

    print("=== Per pressure level ===")
    print(f"{'pressure':<11}{'policy':<12}{'profile':<10}{'completion':<13}"
          f"{'miss rate':<12}{'mean lat':<11}{'p95 lat'}")
    for pressure in sorted(by):
        for policy in ("static", "switching"):
            r = by[pressure].get(policy)
            if r is None:
                continue
            reps = r["reps"]
            comp = statistics.mean(x["completion_rate"] for x in reps)
            miss = statistics.mean(x["miss_rate"] for x in reps)
            lats = [l for x in reps for l in x["latencies"]]
            mean = statistics.mean(lats) if lats else float("nan")
            p95 = (sorted(lats)[int(0.95 * len(lats))] if lats else float("nan"))
            prof = r["profile_chosen"]["name"] if r["profile_chosen"] else "none"
            print(f"{pressure:<11.1f}{policy:<12}{prof:<10}{comp:<13.2f}"
                  f"{miss:<12.2f}{mean:<11.2f}{p95:.2f}")
        print()

    print("=== Paired comparison per pressure level (miss rate, by repetition) ===")
    for pressure in sorted(by):
        s = by[pressure].get("static")
        w = by[pressure].get("switching")
        if not s or not w:
            continue
        a = [x["miss_rate"] for x in s["reps"]]
        b = [x["miss_rate"] for x in w["reps"]]
        n = min(len(a), len(b))
        a, b = a[:n], b[:n]
        diffs = [y - x for x, y in zip(a, b)]
        if all(d == 0 for d in diffs):
            print(f"  {pressure:>5.1f}GB  n={n}  arms identical on every repetition; "
                  f"no test applied")
            continue
        try:
            stat, pval = wilcoxon(b, a)
            eff = rank_biserial(b, a)
            direction = "switching better" if statistics.mean(diffs) < 0 else "static better"
            print(f"  {pressure:>5.1f}GB  n={n}  W={stat:.1f}  p={pval:.4f}  "
                  f"rank-biserial={eff:+.3f}  mean diff={statistics.mean(diffs):+.3f} "
                  f"({direction})")
        except Exception as e:
            print(f"  {pressure:>5.1f}GB  n={n}  test unavailable: {e}")

    print("\n=== Completion rate, which separates the arms where both miss ===")
    for pressure in sorted(by):
        s = by[pressure].get("static")
        w = by[pressure].get("switching")
        if not s or not w:
            continue
        cs = statistics.mean(x["completion_rate"] for x in s["reps"])
        cw = statistics.mean(x["completion_rate"] for x in w["reps"])
        note = ""
        if cs == 0 and cw > 0:
            note = "  <- static fails outright, switching completes"
        print(f"  {pressure:>5.1f}GB  static={cs:.2f}  switching={cw:.2f}{note}")


if __name__ == "__main__":
    main()
