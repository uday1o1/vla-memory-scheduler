"""Is recursive bisection necessary, or would a simpler nested rule do?

Move-optimality follows from nestedness alone, and any priority order over the
layers induces a nested family by taking its first k entries. So the bisection
rule earns its place only if simpler orders fail on the other property that
matters, which is spread: the longest run of consecutive non-resident layers,
since that run is what the streaming pipeline must cover.

This asks whether a simpler construction matches it. If one does, the
contribution is nestedness plus any reasonable order, not this rule.

Candidates are compared on the spread they produce at every residency level,
under the same constraints the real rule obeys: layer 0 is always resident and
the final layer never is.
"""
from __future__ import annotations

import sys
from pathlib import Path as _Path

sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))

from scheduler.paths import bootstrap  # noqa: E402

bootstrap()

from scheduler.placement import max_gap, priority_order  # noqa: E402

TOTAL = 36


def usable(total: int = TOTAL) -> list[int]:
    """Layers eligible for residency: the final layer is never resident."""
    return list(range(total - 1))


def order_bisection(total: int = TOTAL) -> list[int]:
    """The rule under test: recursive bisection of the largest remaining gap."""
    return list(priority_order(total))


def order_sequential(total: int = TOTAL) -> list[int]:
    """Fill from the front. Nested and move-optimal, included as the control."""
    return usable(total)


def order_stride(total: int = TOTAL) -> list[int]:
    """Repeated halving of a stride: 0, then every 16th, every 8th, and so on.

    About as simple as a spread-aware nested order gets.
    """
    out, seen = [], set()
    stride = 1 << (total - 1).bit_length()
    while stride >= 1:
        for i in range(0, total - 1, stride):
            if i not in seen:
                seen.add(i)
                out.append(i)
        stride //= 2
    return out


def order_bitreversal(total: int = TOTAL) -> list[int]:
    """Sort layers by the bit-reversal of their index, a classic low-discrepancy
    ordering that needs no state."""
    bits = max(1, (total - 2).bit_length())

    def rev(i: int) -> int:
        return int(format(i, f"0{bits}b")[::-1], 2)

    return sorted(usable(total), key=lambda i: (rev(i), i))


def order_greedy(total: int = TOTAL) -> list[int]:
    """At each step add whichever layer most reduces the resulting spread.

    Conceptually the simplest statement of the goal, and the most expensive to
    compute. If bisection matches it, bisection is a cheap way to reach the
    same place.
    """
    def profile(idx: list[int]) -> tuple[int, ...]:
        """All gap lengths, longest first. Comparing these lexicographically
        breaks the many ties that comparing only the longest gap leaves, which
        otherwise decides them by index and drifts toward filling from the
        front."""
        s_ = set(idx)
        runs, cur = [], 0
        for i in range(total):
            if i in s_:
                if cur:
                    runs.append(cur)
                cur = 0
            else:
                cur += 1
        if cur:
            runs.append(cur)
        return tuple(sorted(runs, reverse=True))

    chosen = [0]
    remaining = set(usable(total)) - {0}
    while remaining:
        best = min(sorted(remaining), key=lambda c: profile(chosen + [c]))
        chosen.append(best)
        remaining.discard(best)
    return chosen


def order_golden(total: int = TOTAL) -> list[int]:
    """Additive recurrence with the golden ratio, a standard low-discrepancy
    generator, mapped onto layer indices."""
    phi = (5 ** 0.5 - 1) / 2
    n = total - 1
    out, seen = [0], {0}
    x = 0.0
    while len(out) < n:
        x = (x + phi) % 1.0
        i = int(x * n)
        while i in seen:
            i = (i + 1) % n
        seen.add(i)
        out.append(i)
    return out


RULES = {
    "bisection": order_bisection,
    "greedy": order_greedy,
    "stride": order_stride,
    "bitreversal": order_bitreversal,
    "golden": order_golden,
    "sequential": order_sequential,
}


def main() -> None:
    orders = {name: fn(TOTAL) for name, fn in RULES.items()}

    print("=== Validity: nested, layer 0 first, final layer excluded ===")
    for name, o in orders.items():
        ok = (o[0] == 0 and (TOTAL - 1) not in o
              and sorted(o) == list(range(TOTAL - 1)))
        print(f"  {name:<14}{'ok' if ok else 'INVALID'}")
        assert ok, f"{name} is not a valid priority order"
    print("  every rule here is nested by construction, so all are move-optimal")

    ks = range(2, TOTAL - 2)
    print(f"\n=== Spread by residency level (max consecutive non-resident) ===")
    print("  K     " + "".join(f"{n:<14}" for n in RULES))
    totals = {n: 0 for n in RULES}
    worst = {n: 0 for n in RULES}
    for k in ks:
        row = f"  {k:<6}"
        for name in RULES:
            g = max_gap(sorted(orders[name][:k]), TOTAL)
            totals[name] += g
            worst[name] = max(worst[name], g)
            row += f"{g:<14}"
        print(row)

    print("\n=== Spread at the three pre-profiled operating points ===")
    print(f"  {'rule':<14}{'K=16':<8}{'K=24':<8}{'K=33':<8}{'sum'}")
    for name in RULES:
        gs = [max_gap(sorted(orders[name][:k]), TOTAL) for k in (16, 24, 33)]
        print(f"  {name:<14}{gs[0]:<8}{gs[1]:<8}{gs[2]:<8}{sum(gs)}")

    print("\n=== Summary over all residency levels ===")
    print(f"  {'rule':<14}{'mean spread':<14}{'worst spread'}")
    ranked = sorted(RULES, key=lambda n: (totals[n], worst[n]))
    for name in ranked:
        print(f"  {name:<14}{totals[name] / len(list(ks)):<14.2f}{worst[name]}")

    best = ranked[0]
    tied = [n for n in RULES if totals[n] == totals[best] and worst[n] == worst[best]]
    print()
    if tied == ["bisection"] or tied == [best] and best == "bisection":
        print("  Bisection is strictly best: no simpler candidate matches it.")
    elif "bisection" in tied:
        print(f"  Tied at the top: {', '.join(sorted(tied))}.")
        print("  Bisection is not uniquely necessary; any rule in this set would do,")
        print("  and the contribution is nestedness plus a spread-aware order.")
    else:
        print(f"  Bisection is NOT best. {best} beats it, so the rule should change.")


if __name__ == "__main__":
    main()
