"""Can the trace experiment tell its arms apart? A check run before the GPU.

The GPU experiment takes hours, so before running it this asks whether its
design can produce a difference at all, using costs measured earlier rather
than new measurements. It replays the same trace and the same policy
decisions the GPU worker makes (scheduler/trace.py), charging each call its
profile's measured latency plus the cost of any residency change before it.

Per-move costs come from the committed RTX 3090 measurements in
data/adaptive_cost_3090.json: with the upstream demotion mechanism, demoting
eight layers took 5.71s and promoting eight took 0.28s. Demotion with clean
eviction has not been measured; its cost is set to zero here as the
hypothesis under test, and experiments/placement/eviction.py measures it.

Nothing printed here is a result. It is a design check: if two arms come out
identical under these costs, the GPU run cannot separate them either.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scheduler.placement import nested_placement, upstream_placement  # noqa: E402
from scheduler.profiles import PROFILES, deadline_s  # noqa: E402
from scheduler.trace import POLICIES, budget_gb, choose, expand, tenant_fits  # noqa: E402

TOTAL_GB, SAFETY_GB = 24.0, 1.0
PROMOTE_S = 0.28 / 8
DEMOTE_S = {"copy": 5.71 / 8, "clean": 0.0}
PLACE = {"nested": nested_placement, "upstream": upstream_placement}


def arms():
    for p in POLICIES:
        if p == "runtime":
            for placement in PLACE:
                for demotion in DEMOTE_S:
                    yield p, placement, demotion
        elif p == "deadline":
            for placement in PLACE:
                yield p, placement, "copy"
        else:
            yield p, "nested", "copy"


def simulate(policy, placement, demotion, demands, d):
    budgets = [budget_gb(TOTAL_GB, x, SAFETY_GB) for x in demands]
    cur, resident = None, set()
    out = {"miss": 0, "transition_miss": 0, "refused": 0, "transition_s": 0.0}
    for b, demand in zip(budgets, demands):
        nxt = choose(policy, b, cur, budgets)
        target = set(PLACE[placement](nxt.k))
        cost = 0.0
        if cur is not None and nxt.k != cur.k:
            cost = (len(target - resident) * PROMOTE_S
                    + len(resident - target) * DEMOTE_S[demotion])
        resident, cur = target, nxt
        total = nxt.latency_s + cost
        out["transition_s"] += cost
        out["miss"] += total > d
        out["transition_miss"] += total > d >= nxt.latency_s
        out["refused"] += not tenant_fits(TOTAL_GB, nxt.footprint_gb, demand, SAFETY_GB)
    return out


def main():
    d = deadline_s()
    print(f"Deadline {d:.2f}s, profiles "
          + ", ".join(f"{p.name} {p.latency_s:.2f}s" for p in PROFILES))
    for phase_calls in (6, 3, 2, 1):
        table(phase_calls, d)


def table(phase_calls, d):
    demands = expand(phase_calls)
    print(f"\n=== {phase_calls} calls per phase, {len(demands)} calls ===")
    print(f"{'policy':<16}{'placement':<11}{'demotion':<10}{'misses':<9}"
          f"{'from transitions':<18}{'transition s':<14}{'tenant refused'}")
    res = {}
    for a in arms():
        r = simulate(*a, demands, d)
        res[a] = r
        print(f"{a[0]:<16}{a[1]:<11}{a[2]:<10}{r['miss']:<9}{r['transition_miss']:<18}"
              f"{r['transition_s']:<14.2f}{r['refused']}")

    rt = {k[1:]: v for k, v in res.items() if k[0] == "runtime"}
    up_copy, ne_copy = rt[("upstream", "copy")], rt[("nested", "copy")]
    up_clean, ne_clean = rt[("upstream", "clean")], rt[("nested", "clean")]
    print("\nDesign checks:")
    checks = {
        "nested placement changes misses under upstream demotion":
            ne_copy["miss"] != up_copy["miss"],
        "clean eviction changes misses under nested placement":
            ne_clean["miss"] != ne_copy["miss"],
        "runtime switching admits the tenant where startup refuses":
            ne_clean["refused"] < res[("startup", "nested", "copy")]["refused"],
    }
    for name, ok in checks.items():
        print(f"  {'yes' if ok else 'NO ':<4}{name}")
    print("\n  A 'NO' means the GPU run cannot separate those arms under the measured"
          "\n  costs, so that comparison would be uninformative as designed.")


if __name__ == "__main__":
    main()
