"""Build the residency profiles for the machine in use.

The default profiles in scheduler/profiles.py were measured on an RTX 3090.
Latency and footprint both differ on other hardware, and the deadline is
derived from the middle profile's latency, so using them elsewhere would set a
deadline for a machine that is not the one running.

This combines the two characterization runs into the profile file the policy
reads:

    python experiments/characterize/calibrate.py --k-values ...
    python experiments/characterize/vram_per_k.py --k-values ...
    python experiments/characterize/build_profiles.py
    VLA_PROFILES=data/profiles.json python experiments/switching/run.py

Three profiles are chosen from the measured levels: the fastest that was
measured, the slowest, and the one nearest the middle by latency. The
selection rule needs them spaced, and the deadline needs the middle one to sit
between the other two, which the profiles module asserts.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from pathlib import Path as _Path

sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))

from scheduler.paths import RESULTS  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--calibration", type=Path, default=RESULTS / "calibration.json")
    p.add_argument("--vram", type=Path, default=RESULTS / "vram_per_k.json")
    p.add_argument("--output", type=Path, default=RESULTS / "profiles.json")
    p.add_argument("--names", nargs=3, default=["fast", "mid", "compact"])
    args = p.parse_args()

    for f in (args.calibration, args.vram):
        if not f.exists():
            raise SystemExit(
                f"missing {f}. Run calibrate.py and vram_per_k.py on this machine first."
            )

    cal = json.loads(args.calibration.read_text())["calibration"]
    vram = json.loads(args.vram.read_text())["by_k"]

    shared = sorted(set(cal) & set(vram), key=int)
    if len(shared) < 3:
        raise SystemExit(
            f"need at least three residency levels measured by both runs, found {shared}. "
            "Re-run calibrate.py and vram_per_k.py with the same --k-values."
        )

    levels = [
        {"k": int(k),
         "latency_s": float(cal[k]["mean_s"]),
         "footprint_gb": float(vram[k]["process_used_gb"])}
        for k in shared
    ]
    levels.sort(key=lambda e: e["latency_s"])

    # Fastest, nearest the middle by latency, and slowest.
    mid_target = (levels[0]["latency_s"] + levels[-1]["latency_s"]) / 2
    middle = min(levels[1:-1], key=lambda e: abs(e["latency_s"] - mid_target)) \
        if len(levels) > 2 else levels[1]
    chosen = [levels[0], middle, levels[-1]]

    profiles = [
        {"name": n, "k": e["k"], "latency_s": e["latency_s"],
         "footprint_gb": e["footprint_gb"]}
        for n, e in zip(args.names, chosen)
    ]

    print(f"{'name':<10}{'K':<6}{'latency s':<13}{'footprint GB'}")
    for pr in profiles:
        print(f"{pr['name']:<10}{pr['k']:<6}{pr['latency_s']:<13.2f}{pr['footprint_gb']:.2f}")

    deadline = profiles[1]["latency_s"] * 1.15
    meets = [pr["name"] for pr in profiles if pr["latency_s"] <= deadline]
    print(f"\nDeadline from the middle profile times 1.15: {deadline:.2f}s")
    print(f"  meeting it: {', '.join(meets)}")
    if not 0 < len(meets) < len(profiles):
        raise SystemExit(
            "the deadline does not separate these profiles, so a comparison against "
            "static residency would carry no information. Measure levels that are "
            "further apart in latency."
        )
    print("  the deadline separates the profiles, so the comparison is graded")

    args.output.write_text(json.dumps({"profiles": profiles}, indent=2))
    print(f"\nSaved -> {args.output}")
    print(f"Use with:  VLA_PROFILES={args.output} python experiments/switching/run.py")


if __name__ == "__main__":
    main()
