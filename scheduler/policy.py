"""Deadline-and-contention-aware continuous-K residency controller.

Decision cost (cheap): pick target K from an empirical latency(K) calibration
table, adjusted for sensed GPU contention.

Transition cost (expensive, measured separately by
experiments/placement/eviction.py): shedding layers costs about six times more
than restoring the same number (roughly 0.30s against 0.05s per layer on the
primary RTX 3090 host, data/eviction_copy_nested_3090.json), because a shed
layer is copied back to pageable host memory and then pinned, while a restored
layer is one copy from memory already pinned. The copy to pageable memory is
the main cost; pinning adds about 0.43s per eight layers.
Hysteresis exists to avoid paying this cost on every call when it is not
justified by real deadline risk.
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


# Mean latency per K on an RTX 3090 whose host is not recorded: the table
# experiments/contention/run.py uses. At K=16, 24 and 33 these are the values
# DEFAULT_PROFILES in scheduler/profiles.py rounds to two decimals.
# The primary host's calibration, data/calibration_3090.json, agrees within
# 4.4 percent (largest gap at K=33, 6.730s against 7.022s).
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
    OTHER K values would achieve under the same slowdown, assuming the same
    factor at every K. experiments/contention/interaction.py finds instead a
    roughly constant added time (about two seconds at K=16, 24 and 33), so
    the projection overstates the slowdown at low K.
    Falls back to the clean calibration when not yet available (e.g. the
    first few calls before a rolling window fills).

    Both the miss-check and the projection are grounded in observed latency
    rather than a fixed multiplier, because contention slowdown is not known
    ahead of time: sustained slowdowns of 31 and 32 percent were measured on
    two clips, which any static estimate would have to guess at.

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

    # Measured contention on clip d497f01b (data/compute_contention_3090.json):
    # 7.008s -> 9.155s mean latency at K=33 under steady contention, a 31%
    # slowdown. Against the calibrated K=33 latency (7.022s) the factor puts
    # the effective deadline (~6.19s) below every K's clean latency, so
    # min_k_for_deadline falls back to max_k and the deadline-miss override
    # fires: the policy detects the miss even though residency cannot fix it.
    # Compute contention adds roughly the same time at every K
    # (experiments/contention/interaction.py), so no residency level escapes it.
    real_observed_latency = 9.155
    k = decide_residency(state, deadline_s=deadline, gpu_util=0.8,
                          observed_recent_latency=real_observed_latency)
    assert k == 33, f"expected fallback to max_k when no K can meet effective deadline, got {k}"
    print(f"Real contention factor ({real_observed_latency / cal.latency_at(33):.3f}x) -> K={k} "
          "(miss correctly detected, no K can escape it)")

    tight_deadline = cal.latency_at(16)
    state2 = PolicyState(current_k=33, max_k=33)
    k = decide_residency(state2, deadline_s=tight_deadline, gpu_util=0.1)
    print(f"Deadline tight enough to require K=16 -> K={k}")
    assert k == 16, f"expected exact min-K for this deadline, got {k}"

    # K=30 is only 2 layers off the ideal (32), within hysteresis_layers, but
    # its OWN latency (8.761s) already exceeds the 8.075s deadline - the
    # override must fire regardless of the small layer-count delta.
    state3 = PolicyState(current_k=30, max_k=33)
    k = decide_residency(state3, deadline_s=deadline, gpu_util=0.1)
    assert k == 32, f"expected deadline-miss override despite small delta, got {k}"
    print(f"K=30 (delta=2 from ideal=32) already breaches deadline -> K={k} (override fires, hysteresis does not mask a real miss)")

    print("\nAll self-checks passed.")


if __name__ == "__main__":
    demo()
