"""Analysis of the trace experiment.

Two measures, because the policies trade one against the other: the driving
model's deadline-miss rate, with residency changes charged to the call after
them, and how often the tenant is refused its memory. A policy is better only
if it is no worse on either. Mean call latency and the share of calls on which
both parties were served are reported alongside, since a policy can trade one
measure for the other without changing their sum.

Misses are split by cause. A call whose inference alone would have met the
deadline but whose total did not was missed because of the residency change
before it. That count is the direct answer to whether migration cost makes a
better configuration miss deadlines, and it is what the four runtime arms are
compared on as a two-by-two ablation.

Repetitions are paired by clip and compared with the Wilcoxon signed-rank
test. With eight pairs the smallest attainable p is 0.0078, reached only when
every pair moves the same way.
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

from scipy.stats import rankdata, wilcoxon


def rank_biserial(a, b) -> float:
    """Matched-pairs rank-biserial correlation, ties given their mean rank."""
    diffs = [x - y for x, y in zip(a, b) if x != y]
    if not diffs:
        return 0.0
    ranks = rankdata([abs(d) for d in diffs])
    pos = sum(r for r, d in zip(ranks, diffs) if d > 0)
    neg = sum(r for r, d in zip(ranks, diffs) if d < 0)
    return (pos - neg) / (pos + neg)


def per_run(r) -> dict:
    calls, d = r["calls"], r["deadline_s"]
    n = len(r["demands"])
    inferred = [c for c in calls if c["inference_s"] is not None]
    rose = [c for prev, c in zip(calls, calls[1:]) if c["demand_gb"] > prev["demand_gb"]]
    return {
        "calls": len(calls), "planned": n,
        "miss_rate": sum(not c["met_deadline"] for c in calls) / n + (n - len(calls)) / n,
        "transition_misses": sum(c["inference_s"] <= d < c["total_s"] for c in inferred),
        "transition_s": sum(c["transition_s"] for c in calls),
        "switches": sum(bool(c["switched_from"]) for c in calls),
        "refused_rate": sum(not c["tenant_admitted"] for c in calls) / max(1, len(calls)),
        "tenant_wait_s": statistics.median([c["tenant_wait_s"] for c in rose]) if rose else None,
        "oom": sum(c["error"] is not None for c in calls),
        "mean_latency_s": statistics.mean([c["total_s"] for c in inferred]) if inferred else None,
        "both_served": sum(c["met_deadline"] and c["tenant_admitted"] for c in calls) / n,
        "median_inference_s": statistics.median([c["inference_s"] for c in inferred])
                              if inferred else None,
    }


def arm_name(r) -> tuple:
    a = r["args"]
    return a["policy"], a["placement"], a["demotion"], int(a["phase_calls"])


def compare(label, rows, key, a, b):
    """Paired comparison of arm a against arm b on `key`, over shared reps."""
    reps = sorted(set(rows[a]) & set(rows[b]))
    if not reps:
        return
    x = [rows[a][k][key] for k in reps]
    y = [rows[b][k][key] for k in reps]
    diffs = [u - v for u, v in zip(x, y)]
    if all(v == 0 for v in diffs):
        print(f"  {label:<52} identical in all {len(reps)} pairs")
        return
    p = wilcoxon(x, y, zero_method="wilcox").pvalue if any(diffs) else 1.0
    print(f"  {label:<52} {statistics.mean(x):>8.3f} vs {statistics.mean(y):<8.3f} "
          f"n={len(reps)} p={p:.4f} rb={rank_biserial(x, y):+.3f}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, default=Path("data/trace_results.json"))
    args = p.parse_args()
    data = json.loads(args.input.read_text())
    runs = data["runs"]
    print(f"{len(runs)} runs on {data['meta'].get('gpu', 'unknown GPU')}, "
          f"deadline {runs[0]['deadline_s']:.2f}s\n")

    rows = defaultdict(dict)  # arm -> rep -> metrics
    for r in runs:
        if r.get("error"):
            print(f"  run failed: {arm_name(r)} rep {r['rep']}: {r['error']}")
        rows[arm_name(r)][r["rep"]] = per_run(r)

    print(f"{'policy':<15}{'placement':<10}{'demotion':<9}{'L':<4}{'n':<4}{'miss':<8}"
          f"{'from transitions':<18}{'transition s':<14}{'refused':<9}{'tenant wait s':<15}"
          f"{'mean s':<8}{'both':<7}{'OOM'}")
    for arm in sorted(rows):
        m = list(rows[arm].values())
        mean = lambda k: statistics.mean(x[k] for x in m)  # noqa: E731
        waits = [x["tenant_wait_s"] for x in m if x["tenant_wait_s"] is not None]
        print(f"{arm[0]:<15}{arm[1]:<10}{arm[2]:<9}{arm[3]:<4}{len(m):<4}"
              f"{mean('miss_rate'):<8.3f}{mean('transition_misses'):<18.2f}"
              f"{mean('transition_s'):<14.2f}{mean('refused_rate'):<9.3f}"
              f"{(statistics.median(waits) if waits else float('nan')):<15.2f}"
              f"{mean('mean_latency_s'):<8.2f}{mean('both_served'):<7.3f}"
              f"{sum(x['oom'] for x in m)}")

    churns = sorted({a[3] for a in rows})
    for L in churns:
        best = ("runtime", "nested", "clean", L)
        print(f"\n=== {L} calls per phase ===")
        print("Profile switching against the static alternatives:")
        for s in ("startup", "static_fast", "static_compact"):
            # Static arms may have run at one churn rate only. Their latency
            # cannot depend on it, and every churn rate repeats the same
            # pattern, so each demand occupies the same share of calls and
            # their refusal rate is the same at every churn rate too.
            stat = next((a for a in rows if a[0] == s and a[3] == L),
                        next((a for a in rows if a[0] == s), None))
            if stat:
                compare(f"miss rate, runtime nested clean vs {s}", rows, "miss_rate", best, stat)
                compare(f"tenant refused, runtime nested clean vs {s}", rows,
                        "refused_rate", best, stat)
                compare(f"mean latency, runtime nested clean vs {s}", rows,
                        "mean_latency_s", best, stat)
                compare(f"both served, runtime nested clean vs {s}", rows,
                        "both_served", best, stat)
        print("Ablation of the two fixes, misses caused by transitions:")
        rt = lambda pl, de: ("runtime", pl, de, L)  # noqa: E731
        compare("nested vs upstream placement, copy demotion", rows,
                "transition_misses", rt("nested", "copy"), rt("upstream", "copy"))
        compare("clean vs copy demotion, nested placement", rows,
                "transition_misses", rt("nested", "clean"), rt("nested", "copy"))
        compare("nested vs upstream placement, clean demotion", rows,
                "transition_misses", rt("nested", "clean"), rt("upstream", "clean"))
        compare("both fixes vs neither", rows, "transition_misses",
                rt("nested", "clean"), rt("upstream", "copy"))
        print("Changing less often, under the upstream demotion mechanism:")
        compare("deadline vs runtime policy, nested copy, misses", rows, "miss_rate",
                ("deadline", "nested", "copy", L), rt("nested", "copy"))
        compare("deadline vs runtime policy, nested copy, transition misses", rows,
                "transition_misses", ("deadline", "nested", "copy", L), rt("nested", "copy"))
        compare("deadline policy, nested vs upstream, transition misses", rows,
                "transition_misses", ("deadline", "nested", "copy", L),
                ("deadline", "upstream", "copy", L))
        print("Same ablation on total transition seconds:")
        compare("nested vs upstream placement, copy demotion", rows,
                "transition_s", rt("nested", "copy"), rt("upstream", "copy"))
        compare("clean vs copy demotion, nested placement", rows,
                "transition_s", rt("nested", "clean"), rt("nested", "copy"))
        compare("nested vs upstream placement, clean demotion", rows,
                "transition_s", rt("nested", "clean"), rt("upstream", "clean"))

    # Deadline sensitivity. The runtime policy never consults the deadline, so
    # its recorded calls can be scored against any deadline after the fact. The
    # deadline fixed before the run (mid profile x 1.15) stays the primary
    # result; this shows how far the ablation's conclusion depends on it. All
    # four arms choose the same profiles, so the gaps between them are what
    # residency changes cost.
    mid = runs[0]["deadline_s"] / 1.15
    mults = (1.05, 1.10, 1.15, 1.20, 1.30)
    misses = lambda r, d: sum(c["total_s"] is None or c["total_s"] > d for c in r["calls"])  # noqa: E731
    print("\n=== Deadline sensitivity, runtime arms, misses per run ===")
    print(f"{'placement':<10}{'demotion':<9}{'L':<4}" + "".join(f"x{m:<7.2f}" for m in mults))
    for arm in sorted(a for a in rows if a[0] == "runtime"):
        rs = [r for r in runs if arm_name(r) == arm]
        cells = [statistics.mean(misses(r, mid * m) for r in rs) for m in mults]
        print(f"{arm[1]:<10}{arm[2]:<9}{arm[3]:<4}" + "".join(f"{v:<8.2f}" for v in cells))
    for L in churns:
        for m in (1.05, 1.10):
            tight = defaultdict(dict)
            for r in runs:
                tight[arm_name(r)][r["rep"]] = {"m": misses(r, mid * m)}
            compare(f"x{m:.2f}, L={L}, nested vs evenly spaced, host copy", tight, "m",
                    ("runtime", "nested", "clean", L), ("runtime", "upstream", "clean", L))

    # The price of nesting: a nested order fixes one priority order for every
    # K, so at a given K it may place layers worse than a rule free to choose
    # per K. Steady-state inference at the same K, same rep, both placements,
    # calls straight after a change excluded.
    print("\n=== Price of nesting: steady inference at equal K, nested vs upstream ===")
    steady = defaultdict(lambda: defaultdict(list))  # (placement, k) -> rep -> times
    for r in runs:
        pol, pl, de, L = arm_name(r)
        if pol != "runtime":
            continue
        for c in r["calls"]:
            if c["inference_s"] is not None and not c["switched_from"]:
                steady[(pl, c["k"])][r["rep"]].append(c["inference_s"])
    for k in sorted({k for _, k in steady}):
        n, u = steady[("nested", k)], steady[("upstream", k)]
        reps = sorted(set(n) & set(u))
        if len(reps) < 2:
            continue
        x = [statistics.median(n[i]) for i in reps]
        y = [statistics.median(u[i]) for i in reps]
        diff = 100 * (statistics.mean(x) - statistics.mean(y)) / statistics.mean(y)
        p_ = wilcoxon(x, y).pvalue if any(a != b for a, b in zip(x, y)) else 1.0
        print(f"  K={k:<4}nested {statistics.mean(x):.3f}s  upstream {statistics.mean(y):.3f}s  "
              f"{diff:+.2f}%  n={len(reps)} p={p_:.4f}")

    # Position check: a slower first-run arm would show here.
    pos = [(r["position"], per_run(r)["median_inference_s"]) for r in runs
           if per_run(r)["median_inference_s"] and arm_name(r)[0] == "static_fast"]
    if len(pos) > 3:
        first = [v for p_, v in pos if p_ == 0]
        rest = [v for p_, v in pos if p_ > 0]
        if first and rest:
            print(f"\nstatic_fast median inference when run first {statistics.mean(first):.3f}s, "
                  f"otherwise {statistics.mean(rest):.3f}s")


if __name__ == "__main__":
    main()
