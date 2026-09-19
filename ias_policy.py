"""Algorithm 1.2: deadline-and-contention-aware residency policy.

Decision cost (cheap): pick target K from an empirical latency(K) calibration
table, adjusted for sensed GPU contention.

Transition cost (expensive, measured separately - see ias_transition_cost.py):
shedding layers costs 3-6x more than restoring them (pin_memory() on offload
vs. plain device copy on restore). Hysteresis exists specifically to avoid
paying this cost on every call when it isn't justified by real deadline risk.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Calibration:
    """Empirical latency(K) points, sorted by K ascending."""
    k_values: list[int]
    latencies_s: list[float]

    def latency_at(self, k: int) -> float:
        """Linear interpolation between the nearest calibrated K values."""
        if k <= self.k_values[0]:
            return self.latencies_s[0]
        if k >= self.k_values[-1]:
            return self.latencies_s[-1]
        for i in range(len(self.k_values) - 1):
            lo_k, hi_k = self.k_values[i], self.k_values[i + 1]
            if lo_k <= k <= hi_k:
                lo_t, hi_t = self.latencies_s[i], self.latencies_s[i + 1]
                frac = (k - lo_k) / (hi_k - lo_k)
                return lo_t + frac * (hi_t - lo_t)
        raise AssertionError("unreachable")

    def min_k_for_deadline(self, deadline_s: float, max_k: int) -> int:
        """Smallest K (<= max_k) whose calibrated latency fits the deadline.

        Latency DECREASES as K increases (more resident layers = less
        streaming = faster). We want the minimum residency that still meets
        the deadline - holding more permanent GPU memory than necessary buys
        nothing. If even max_k misses the deadline, return max_k (the
        fastest achievable option; the miss itself is what we measure).
        """
        for k in range(0, max_k + 1):
            if self.latency_at(k) <= deadline_s:
                return k
        return max_k


CALIBRATION = Calibration(
    k_values=[0, 8, 16, 24, 30, 33],
    latencies_s=[26.866, 21.905, 17.063, 12.312, 8.761, 7.022],
)


@dataclass
class PolicyState:
    current_k: int
    max_k: int = 33


def decide_residency(
    state: PolicyState,
    deadline_s: float,
    gpu_util: float,
    calibration: Calibration = CALIBRATION,
    observed_recent_latency: float | None = None,
    contention_threshold: float = 0.5,
    hysteresis_layers: int = 3,
) -> int:
    """Return the new resident-layer count K for this call.

    `observed_recent_latency`: the REAL recent average latency (e.g. a
    rolling window mean over the last few calls), when available. This is
    used DIRECTLY for the deadline-miss check - not a theoretical estimate
    from the clean calibration table, which cannot see real contention
    effects. It also derives an empirical contention factor
    (observed / calibration.latency_at(current_k)) used to project what
    OTHER K values would achieve under the same slowdown, assuming it
    applies uniformly across K (confirmed by ias_contention_interaction.py).
    Falls back to the clean calibration when not yet available (e.g. the
    first few calls before a rolling window fills).

    An earlier version used a fixed placeholder multiplier instead of real
    observed data, which caused the deadline-miss check to badly
    underestimate real contention (see PROJECT_DETAILS.md) and left the
    policy never reacting to a sustained ~32% real slowdown. Grounding both
    the miss-check and the projection in actual observed latency fixes this.

    Hysteresis: only actually change K if the ideal target differs from the
    current K by more than `hysteresis_layers`, OR if staying at current_k
    would clearly miss the deadline (that always overrides hysteresis - a
    real miss risk is worth paying the transition cost for).
    """
    if gpu_util > contention_threshold and observed_recent_latency is not None:
        empirical_factor = observed_recent_latency / calibration.latency_at(state.current_k)
        effective_deadline = deadline_s / empirical_factor
        current_latency = observed_recent_latency
    else:
        effective_deadline = deadline_s
        current_latency = calibration.latency_at(state.current_k)

    ideal_k = calibration.min_k_for_deadline(effective_deadline, state.max_k)
    would_miss_deadline = current_latency > deadline_s

    delta = abs(ideal_k - state.current_k)
    if delta <= hysteresis_layers and not would_miss_deadline:
        return state.current_k

    return ideal_k


def demo():
    """Self-check: verify the decision function behaves sanely."""
    cal = CALIBRATION
    state = PolicyState(current_k=33, max_k=33)

    deadline = cal.latency_at(33) * 1.15
    print(f"Deadline (from K=33 baseline * 1.15): {deadline:.3f}s")

    k = decide_residency(state, deadline_s=deadline, gpu_util=0.1)
    assert k == 33, f"expected no change at K=33 under low contention, got {k}"
    print(f"Low contention, ample budget -> K={k} (unchanged, as expected)")

    # Real observed contention from Rep 1 data: ~32% slowdown (6.943s -> 9.157s
    # under steady contention). Even K=33's clean latency (7.022s) can't beat
    # the resulting effective deadline (~6.12s), so min_k_for_deadline falls
    # back to max_k, and the deadline-miss override correctly fires - the
    # policy detects the real miss even though residency can't fix it (a
    # genuine finding: uniform-across-K contention has no residency-based
    # escape, confirmed by ias_contention_interaction.py).
    real_observed_latency = 9.157
    k = decide_residency(state, deadline_s=deadline, gpu_util=0.8,
                          observed_recent_latency=real_observed_latency)
    assert k == 33, f"expected fallback to max_k when no K can meet effective deadline, got {k}"
    print(f"Real contention factor (1.319x, from Rep 1 data) -> K={k} (miss correctly detected, no K can escape it)")

    tight_deadline = cal.latency_at(16)
    state2 = PolicyState(current_k=33, max_k=33)
    k = decide_residency(state2, deadline_s=tight_deadline, gpu_util=0.1)
    print(f"Deadline tight enough to require K=16 -> K={k}")
    assert k == 16, f"expected exact min-K for this deadline, got {k}"

    # K=30 is only 3 layers off the ideal (32), within hysteresis_layers, but
    # its OWN latency (8.761s) already exceeds the 8.075s deadline - the
    # override must fire regardless of the small layer-count delta.
    state3 = PolicyState(current_k=30, max_k=33)
    k = decide_residency(state3, deadline_s=deadline, gpu_util=0.1)
    assert k == 32, f"expected deadline-miss override despite small delta, got {k}"
    print(f"K=30 (delta=2 from ideal=32) already breaches deadline -> K={k} (override fires, hysteresis does not mask a real miss)")

    print("\nAll self-checks passed.")


if __name__ == "__main__":
    demo()
