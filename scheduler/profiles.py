"""Pre-profiled residency configurations and the profile-switching policy.

Profiles, rather than an arbitrary residency count re-derived on every call,
for two reasons:

  1. Measured transition cost. Changing residency costs 0.43 to 4.97 seconds
     in the isolated demotion runs (0.27 to 0.71 with a host copy kept;
     medians averaged over both orders, data/eviction_*_3090.json) and 0.43
     to 7.51 seconds per change inside the trace experiment (0.28 to 0.94
     with a host copy kept, data/trace_results_3090.json), against 6.4 to
     17.7 seconds of inference per call in that experiment, so a policy free
     to move to any K on any call can spend a large fraction of its budget
     reconfiguring. A small fixed profile set bounds both how often
     transitions happen and how large each one is.

  2. A deadline anchored to the fastest configuration leaves one lever,
     lowering K, and that lever only makes latency worse, so it cannot
     separate an adaptive policy from a static one. The deadline here is
     anchored to the middle profile instead (see below).

Profiles are spaced across the measured latency and footprint curves. The
selection rule picks the fastest profile that fits a memory budget with
margin. The launch-time experiments (experiments/switching,
experiments/dynamic) give it free GPU memory, which torch.cuda.mem_get_info
reports across processes and which needs no elevated privilege; the trace
experiment gives it the memory the tenant leaves (scheduler/trace.py).
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Profile:
    name: str
    k: int
    latency_s: float       # measured, no contention
    footprint_gb: float    # measured, peak during real inference

    def __str__(self) -> str:
        return f"{self.name}(K={self.k}, {self.latency_s:.2f}s, {self.footprint_gb:.2f}GB)"


def load_profiles(path: str | Path) -> tuple[Profile, ...]:
    """Read profiles measured on the machine in use.

    The file holds a list of objects with name, k, latency_s and
    footprint_gb, either bare or under "profiles" as
    experiments/characterize/build_profiles.py writes it. Profiles are
    returned fastest first, which is the order the selection rule expects.
    """
    entries = json.loads(Path(path).read_text())
    if isinstance(entries, dict):
        entries = entries["profiles"]
    built = tuple(
        Profile(e["name"], int(e["k"]), float(e["latency_s"]), float(e["footprint_gb"]))
        for e in entries
    )
    return tuple(sorted(built, key=lambda p: p.latency_s))


# Built-in defaults: the RTX 3090 profiles the switching and dynamic
# experiments used (recorded in data/switch_results_3090.json and
# data/dynamic_results_3090.json, deadline 14.1565s). The committed RTX 3090
# profiles from the characterization runs are data/profiles_3090.json, which
# the trace experiment uses. Latency and footprint both differ on other
# hardware, and the deadline is derived from them. Point VLA_PROFILES at a
# file from build_profiles.py to use profiles measured on the machine in use.
DEFAULT_PROFILES: tuple[Profile, ...] = (
    Profile("fast", 33, 7.02, 17.24),
    Profile("mid", 24, 12.31, 14.14),
    Profile("compact", 16, 17.06, 11.28),
)

# The deadline is anchored to the MIDDLE profile, not the fastest. Anchored to
# the fastest, every profile below it misses by construction, so both arms
# miss and the comparison carries no information. Anchored here, `fast` and
# `mid` meet the deadline and `compact` does not, so the choice of profile
# decides the outcome.
PROFILES: tuple[Profile, ...] = (
    load_profiles(os.environ["VLA_PROFILES"])
    if os.environ.get("VLA_PROFILES") else DEFAULT_PROFILES
)

DEADLINE_ANCHOR = "mid"
DEADLINE_MARGIN = 1.15


def deadline_s(profiles: tuple[Profile, ...] = PROFILES) -> float:
    anchor = next(p for p in profiles if p.name == DEADLINE_ANCHOR)
    return anchor.latency_s * DEADLINE_MARGIN


def free_gb(device_idx: int = 0) -> float:
    """Free GPU memory in GB. Imported lazily so the selection logic above
    stays unit-testable on a machine without CUDA."""
    import torch
    free, _ = torch.cuda.mem_get_info(device_idx)
    return free / (1024 ** 3)


def select_profile(available_gb: float, already_held_gb: float = 0.0,
                   safety_gb: float = 1.0,
                   profiles: tuple[Profile, ...] = PROFILES) -> Profile | None:
    """Fastest profile whose footprint fits in what we can actually reach.

    `already_held_gb` is memory this process has already allocated, which is
    reachable by us even though it is not reported free. Without it the policy
    would refuse to upgrade after having allocated, since its own allocation
    looks like unavailable memory.
    """
    budget = available_gb + already_held_gb - safety_gb
    for p in profiles:  # ordered fastest first
        if p.footprint_gb <= budget:
            return p
    return None


def profile_for_k(k: int, profiles: tuple[Profile, ...] = PROFILES) -> Profile | None:
    return next((p for p in profiles if p.k == k), None)


def _selftest():
    print("Profiles:")
    for p in PROFILES:
        print(f"  {p}")
    d = deadline_s()
    print(f"\nDeadline: {d:.3f}s (anchor '{DEADLINE_ANCHOR}' x {DEADLINE_MARGIN})")

    print("\nWhich profiles meet the deadline:")
    for p in PROFILES:
        print(f"  {p.name:<9} {p.latency_s:>6.2f}s  "
              f"{'meets' if p.latency_s <= d else 'MISSES'}")
    meeting = [p for p in PROFILES if p.latency_s <= d]
    assert 0 < len(meeting) < len(PROFILES), (
        "deadline must separate the profiles; if all or none meet it the "
        "comparison is degenerate")
    print(f"  -> {len(meeting)} of {len(PROFILES)} meet it, so the deadline discriminates")

    print("\nSelection against available memory (nothing held yet):")
    for avail in (24.0, 19.6, 16.6, 13.6, 10.0, 5.0):
        sel = select_profile(avail)
        print(f"  {avail:>5.1f}GB free -> {sel.name if sel else 'NONE FITS'}")

    assert select_profile(24.0).name == "fast"
    assert select_profile(16.6).name == "mid"
    assert select_profile(13.6).name == "compact"
    assert select_profile(5.0) is None

    # A process already holding `compact` should be able to upgrade when the
    # competitor releases, since its own allocation is reachable.
    held = PROFILES[2].footprint_gb
    up = select_profile(available_gb=7.0, already_held_gb=held)
    print(f"\n  holding compact ({held}GB) with 7.0GB free -> {up.name}")
    assert up.name == "fast", "must count already-held memory when upgrading"

    print("\nAll self-checks passed.")


if __name__ == "__main__":
    _selftest()
