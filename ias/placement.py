"""Nested residency placement.

The upstream rule (`interleaved_placement`) spreads K resident layers evenly
across the layer stack. That is correct for a one-shot offline choice, but it
is not nested: the K-layer set is not a subset of the K+1-layer set, so
changing K at runtime reshuffles layers that did not need to move. Measured
across a representative adaptation path it costs about 3x more layer
movements than the change in K requires, and movement is expensive
(roughly 0.35s per layer shed, 0.064s per layer restored).

A nested rule fixes a single priority order over layers and takes the top K.
Every K-set is then a subset of every larger one, so a transition moves
exactly |delta K| layers. The order is built by recursive bisection, so each
prefix is still spread across the stack rather than clustered, preserving
the property the interleaved rule was designed for.
"""
from __future__ import annotations

import heapq
from functools import lru_cache


@lru_cache(maxsize=None)
def priority_order(total: int = 36) -> tuple[int, ...]:
    """Layer indices in descending residency priority.

    Matches the upstream rule's two structural choices: layer 0 is always
    resident first, and the final layer is never resident.
    """
    usable_end = total - 1  # final layer excluded, as upstream does
    order = [0]
    seen = {0}
    # max-heap on interval length; bisect the widest remaining gap each time
    heap = [(-(usable_end - 0), 0, usable_end)]
    while len(order) < usable_end:
        if not heap:
            break
        _, s, e = heapq.heappop(heap)
        mid = (s + e) // 2
        if mid in seen or mid <= s or mid >= e:
            # interval exhausted; try to find any unused index inside it
            cand = next((i for i in range(s + 1, e) if i not in seen), None)
            if cand is None:
                continue
            mid = cand
        order.append(mid)
        seen.add(mid)
        if mid - s > 1:
            heapq.heappush(heap, (-(mid - s), s, mid))
        if e - mid > 1:
            heapq.heappush(heap, (-(e - mid), mid, e))
    return tuple(order)


def nested_placement(k: int, total: int = 36) -> list[int]:
    """Resident indices for residency count k, nested across k."""
    if k <= 0:
        return []
    return sorted(priority_order(total)[:k])


def max_gap(indices: list[int], total: int = 36) -> int:
    """Largest run of consecutive non-resident layers. Lower is better spread."""
    if not indices:
        return total
    s = set(indices)
    gap = best = 0
    for i in range(total):
        if i in s:
            gap = 0
        else:
            gap += 1
            best = max(best, gap)
    return best


def _selftest():
    total = 36
    print("=== Nesting property ===")
    nested_ok = True
    for k in range(1, total - 1):
        a, b = set(nested_placement(k, total)), set(nested_placement(k + 1, total))
        if not a.issubset(b):
            nested_ok = False
            print(f"  VIOLATION at k={k}: {sorted(a - b)} dropped")
    print(f"  nested for all k: {nested_ok}")
    assert nested_ok, "nested_placement must be nested"

    print("\n=== Structural parity with upstream ===")
    for k in (1, 8, 16, 24, 33):
        idx = nested_placement(k, total)
        assert 0 in idx, f"layer 0 must be resident at k={k}"
        assert (total - 1) not in idx, f"final layer must never be resident at k={k}"
        assert len(idx) == k, f"expected {k} layers, got {len(idx)}"
    print("  layer 0 always resident, final layer never resident, exact count: ok")

    print("\n=== Spread quality (max consecutive non-resident run; lower is better) ===")
    # Only a genuinely absent upstream package may skip this comparison. Any
    # other failure is a bug here and must not masquerade as a missing
    # dependency, which would let the self-check pass having compared nothing.
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from ias.paths import bootstrap
    bootstrap()
    try:
        from alpamayo_memopt.profiler import interleaved_placement
        have_upstream = True
    except ModuleNotFoundError:
        have_upstream = False

    if have_upstream:
        print(f"  {'K':<6}{'nested':<10}{'upstream':<10}{'moves nested':<15}{'moves upstream'}")
        prev_n = prev_u = None
        for k in (8, 16, 20, 24, 30, 33):
            n = nested_placement(k, total)
            u = interleaved_placement(k, total)
            mn = mu = "-"
            if prev_n is not None:
                sn, su = set(n), set(u)
                mn = len(set(prev_n) - sn) + len(sn - set(prev_n))
                mu = len(set(prev_u) - su) + len(su - set(prev_u))
            print(f"  {k:<6}{max_gap(n, total):<10}{max_gap(u, total):<10}{str(mn):<15}{mu}")
            prev_n, prev_u = n, u
    else:
        print(f"  {'K':<6}{'nested max_gap'}")
        for k in (8, 16, 20, 24, 30, 33):
            print(f"  {k:<6}{max_gap(nested_placement(k, total), total)}")
        print("  (upstream rule unavailable here; run on the GPU host to compare)")

    print("\nAll self-checks passed.")


if __name__ == "__main__":
    _selftest()
