"""A co-located tenant whose memory demand changes, and the policies that
respond to it.

The setting is a GPU shared between the driving model and another workload,
for instance a perception model that is loaded and unloaded as the driving
context changes. An orchestrator knows the other workload's demand and gives
the driving model the remainder as its budget. GPU memory cannot be taken from
a process that holds it, so the tenant only gets its memory if the driving
model gives it up.

The trace is indexed by call, not by time. Every arm then faces the same
sequence of demands at the same points in its work, regardless of how fast it
runs, which is what lets arms be compared call by call.

Policies:
  static_fast     the upstream offline choice, the largest profile, never
                  changes. Ignores the budget, so the tenant is refused
                  whenever it does not fit beside it.
  static_compact  the profile that fits the worst demand in the trace, never
                  changes. Always admits the tenant, always pays for it.
  startup         chooses from the budget at the first call and holds it,
                  which is what reading free memory once at launch gives.
  runtime         re-chooses from the budget before every call, taking the
                  fastest profile that fits. This is the profile switching
                  the comparison is about.
  deadline        re-chooses before every call, taking the smallest profile
                  that still meets the deadline, and the fastest that fits
                  only when none does. Speed beyond the deadline buys
                  nothing, so it holds less memory and changes less often.

This module has no GPU dependency, so the decisions it makes can be checked
anywhere and are shared unchanged by the simulator and the GPU worker.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scheduler.profiles import PROFILES, Profile, deadline_s, select_profile  # noqa: E402

POLICIES = ("static_fast", "static_compact", "startup", "runtime", "deadline")

# Tenant demand in GB, one entry per phase. Starts under pressure so that
# choosing at startup differs from the static offline choice, then makes
# every single-step change in both directions and the jump across two in
# both directions, so shedding and restoring are each exercised at both sizes.
PATTERN: tuple[float, ...] = (8.0, 11.0, 8.0, 0.0, 8.0, 0.0, 11.0, 0.0)
CALLS = 48


def expand(phase_calls: int = 6, pattern=PATTERN, calls: int = CALLS) -> list[float]:
    """Demand per call. `phase_calls` is how many calls each phase lasts, so
    it sets the churn rate; the pattern repeats to fill `calls`, keeping the
    amount of work fixed while changes become more frequent."""
    one = [d for d in pattern for _ in range(phase_calls)]
    return (one * (calls // len(one) + 1))[:calls]


def budget_gb(total_gb: float, demand_gb: float, safety_gb: float) -> float:
    return total_gb - demand_gb - safety_gb


def choose(policy: str, budget: float, current: Profile | None,
           trace_budgets: list[float], profiles: tuple[Profile, ...] = PROFILES,
           deadline: float | None = None) -> Profile | None:
    """The profile `policy` runs the next call on, or None if nothing fits."""
    if policy == "static_fast":
        return profiles[0]
    if policy == "static_compact":
        return select_profile(min(trace_budgets), safety_gb=0.0, profiles=profiles)
    if policy == "startup":
        return current or select_profile(budget, safety_gb=0.0, profiles=profiles)
    if policy == "runtime":
        return select_profile(budget, safety_gb=0.0, profiles=profiles)
    if policy == "deadline":
        d = deadline if deadline is not None else deadline_s(profiles)
        enough = [p for p in profiles if p.latency_s <= d and p.footprint_gb <= budget]
        if enough:
            return min(enough, key=lambda p: p.footprint_gb)
        return select_profile(budget, safety_gb=0.0, profiles=profiles)
    raise ValueError(policy)


def tenant_fits(total_gb: float, held_gb: float, demand_gb: float,
                safety_gb: float) -> bool:
    """What the orchestrator expects. The GPU worker checks it by allocating."""
    return demand_gb == 0 or held_gb + demand_gb + safety_gb <= total_gb


def check_design(total_gb: float, safety_gb: float, phase_calls: int = 6,
                 profiles: tuple[Profile, ...] = PROFILES) -> list[str]:
    """Reasons the trace cannot discriminate between policies on this card
    with these profiles, or an empty list. Footprints differ between
    machines, so this must hold for the profiles actually in use."""
    demands = expand(phase_calls)
    budgets = [budget_gb(total_gb, d, safety_gb) for d in demands]
    runs, problems = {}, []
    for p in POLICIES:
        cur, seq = None, []
        for b in budgets:
            cur = choose(p, b, cur, budgets, profiles)
            if cur is None:
                return [f"{p}: no profile fits a budget of {b:.2f}GB"]
            seq.append(cur)
        runs[p] = seq
    if len({tuple(x.k for x in v) for v in runs.values()}) < len(POLICIES):
        problems.append("two policies make identical choices on this trace")
    rt = [x.k for x in runs["runtime"]]
    steps = {(a, b) for a, b in zip(rt, rt[1:]) if a != b}
    ks = [p.k for p in profiles]
    want = {(a, b) for a in ks for b in ks if a != b}
    if steps != want:
        problems.append(f"runtime switching never makes transitions {sorted(want - steps)}")
    if any(not tenant_fits(total_gb, x.footprint_gb, d, safety_gb)
           for x, d in zip(runs["runtime"], demands)):
        problems.append("runtime switching exceeds its budget")
    sw = lambda v: sum(a.k != b.k for a, b in zip(v, v[1:]))  # noqa: E731
    if not sw(runs["deadline"]) < sw(runs["runtime"]):
        problems.append("the deadline policy does not change less often than runtime")
    return problems


def _selftest() -> None:
    total, safety = 24.0, 1.0
    demands = expand()
    budgets = [budget_gb(total, d, safety) for d in demands]
    print(f"{len(demands)} calls, demands {sorted(set(demands))} GB")
    for policy in POLICIES:
        cur, seq, refused = None, [], 0
        for b, d in zip(budgets, demands):
            cur = choose(policy, b, cur, budgets)
            seq.append(cur.name[0])
            refused += not tenant_fits(total, cur.footprint_gb, d, safety)
        print(f"  {policy:<15}{''.join(seq)}  tenant refused on {refused} calls")
    for phase_calls in (6, 2, 1):
        problems = check_design(total, safety, phase_calls)
        assert not problems, problems
    print("All self-checks passed.")


if __name__ == "__main__":
    _selftest()
