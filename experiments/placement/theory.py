"""Move-optimality of nested placement, and an ablation isolating what matters.

Optimality argument. A residency transition from set A (|A| = K) to set B
(|B| = K') must physically move every layer in the symmetric difference
A delta B, so its cost is |A \\ B| + |B \\ A|. Since |A| = K and |B| = K',

    |A delta B| = |A| + |B| - 2|A intersect B| = K + K' - 2|A intersect B|

and |A intersect B| <= min(K, K'), giving

    |A delta B| >= K + K' - 2 min(K, K') = |K' - K|

with equality exactly when A intersect B = min(K, K'), i.e. when one set
contains the other. A placement family is nested precisely when that holds
for every pair, so nested families achieve the lower bound on every
transition and non-nested families strictly exceed it on at least one.

This is combinatorial: it holds for any model, any layer size, any GPU. The
hardware only determines the seconds per move, not the move count.

The ablation asks a separate question: is the benefit from nesting alone, or
does our specific bisection order matter? Comparing against a deliberately
poor nested order (sequential prefix) separates the two. Sequential is
equally nested, so if nesting were the whole story it would perform
identically; if spread also matters, it will show worse layer distribution.
"""
from __future__ import annotations

import sys
from itertools import combinations

from pathlib import Path as _Path  # noqa: E402
sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))

from ias.paths import bootstrap  # noqa: E402

bootstrap()
from ias.placement import nested_placement, max_gap, priority_order  # noqa: E402

TOTAL = 36


def sequential_placement(k: int, total: int = TOTAL) -> list[int]:
    """A deliberately poor but still nested order: layers 0..k-1.

    Nested by construction, so it isolates nesting from spread quality.
    """
    return list(range(min(k, total - 1)))


def verify_lower_bound(placement_fn, name: str, total: int = TOTAL) -> bool:
    """Check the |K' - K| bound empirically over all K pairs."""
    optimal = True
    worst = (0, 0, 0, 0)
    for k1, k2 in combinations(range(1, total - 1), 2):
        a, b = set(placement_fn(k1, total)), set(placement_fn(k2, total))
        moves = len(a - b) + len(b - a)
        bound = abs(k2 - k1)
        if moves > bound:
            optimal = False
            if moves - bound > worst[3] - worst[2]:
                worst = (k1, k2, bound, moves)
    if optimal:
        print(f"  {name:<14} achieves the |dK| lower bound on every transition")
    else:
        k1, k2, bound, moves = worst
        print(f"  {name:<14} EXCEEDS the bound; worst case K={k1}->{k2} "
              f"needs {moves} moves where {bound} suffice ({moves / bound:.1f}x)")
    return optimal


def main():
    print("=== Move-optimality: does each rule achieve the |dK| lower bound? ===")
    verify_lower_bound(nested_placement, "nested")
    verify_lower_bound(sequential_placement, "sequential")
    # Only an absent upstream package may skip the comparison; any other
    # failure is a bug here and must surface rather than be reported as a
    # missing dependency.
    try:
        from alpamayo_memopt.profiler import interleaved_placement
    except ModuleNotFoundError:
        have_upstream = False
        print("  upstream       (unavailable here; run on the GPU host)")
    else:
        verify_lower_bound(interleaved_placement, "upstream")
        have_upstream = True

    print("\n=== Ablation: is the benefit from nesting alone, or from the order? ===")
    print("  Both nested and sequential are nested, so both are move-optimal.")
    print("  Spread quality separates them (max consecutive non-resident run):")
    header = f"  {'K':<6}{'nested':<12}{'sequential':<14}"
    if have_upstream:
        header += "upstream"
    print(header)
    for k in (8, 16, 20, 24, 30, 33):
        row = f"  {k:<6}{max_gap(nested_placement(k), TOTAL):<12}" \
              f"{max_gap(sequential_placement(k), TOTAL):<14}"
        if have_upstream:
            row += str(max_gap(interleaved_placement(k, TOTAL), TOTAL))
        print(row)

    print("\n  If sequential shows worse spread but equal move counts, then nesting")
    print("  and spread are independent properties and the contribution is a rule")
    print("  that achieves both, not merely nesting.")

    print(f"\n=== Priority order (first 12) ===")
    print(f"  {list(priority_order(TOTAL))[:12]}")


if __name__ == "__main__":
    main()
