"""Primary experiment: static residency versus contention-aware profile switching.

This is the comparison the professor asked for, with deadline-miss rate as the
headline metric under a controlled competing GPU workload.

Memory pressure rather than compute contention is the regime under test.
Compute contention was measured to slow every residency level proportionally
(0.988x/1.005x/1.031x at K=16/24/33), leaving residency with no lever against
it, so a static-versus-switching comparison under compute contention alone is
degenerate by construction. That is the exact failure mode that invalidated
the earlier design. Memory pressure is where the residency decision changes
the outcome. The compute-contention result is retained and reported as a
negative boundary on where the mechanism applies.

This driver holds the pressure itself and starts each model run as a
subprocess, since whoever allocates first keeps the memory.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, "/root")
from ias_profiles import PROFILES, deadline_s, select_profile

WORKER = "/root/ias_switch_worker.py"
PY = "/venv/main/bin/python"
CLIPS = [
    "d497f01b-4f68-4c27-9a6c-55872a1d6bd6",
    "441057af-5c65-4d8e-993d-713090072248",
    "59aba96d-8920-4e49-8edb-c08bd800edf1",
    "8825b1fa-0abf-4a28-b444-25fef71cadcb",
    "89e4e404-80ea-42e2-bf8e-8395af5ab111",
    "7d109673-d967-4b02-93c5-6d2d25d964d0",
]
PRESSURES = [0.0, 4.0, 7.0, 10.0]


def hold(gb: float):
    if gb <= 0:
        return None
    t = torch.empty(int(gb * (1024 ** 3) / 4), dtype=torch.float32, device="cuda:0")
    t.fill_(1.0)
    torch.cuda.synchronize()
    return t


def calibrate(safety_gb: float):
    """Verify what each pressure level actually admits.

    Checks the capacity arithmetic against real allocator behavior before
    committing to these levels, since the allocator's reservation behavior
    does not always match naive math.
    """
    from ias_profiles import free_gb
    print("=== Calibration: what does each pressure level admit? ===")
    print(f"{'pressure GB':<14}{'free GB':<12}{'admits'}")
    admits = {}
    for gb in PRESSURES:
        blk = hold(gb)
        f = free_gb()
        sel = select_profile(f, safety_gb=safety_gb)
        admits[gb] = sel.name if sel else "NONE"
        print(f"{gb:<14.1f}{f:<12.2f}{admits[gb]}")
        del blk
        torch.cuda.empty_cache()
        time.sleep(1.0)
    distinct = len(set(admits.values()))
    print(f"\n  distinct outcomes across pressure levels: {distinct}")
    if distinct < 2:
        print("  WARNING: every pressure level admits the same profile, so the")
        print("  switching arm can never differ from the static arm. Adjust the")
        print("  pressure levels before running.")
    print()
    return admits


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--reps", type=int, default=8)
    p.add_argument("--n-calls", type=int, default=20)
    p.add_argument("--placement", choices=["nested", "upstream"], default="nested")
    p.add_argument("--outdir", type=Path, default=Path("/root/switch"))
    p.add_argument("--output", type=Path, default=Path("/root/switch_results.json"))
    p.add_argument("--safety-gb", type=float, default=1.0)
    p.add_argument("--calibrate-only", action="store_true")
    args = p.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    torch.cuda.set_device(0)
    d = deadline_s()
    print(f"Deadline {d:.3f}s, anchored to the middle profile so that it separates")
    print("the profiles. Anchoring to the fastest profile is what made the earlier")
    print("design untestable: every other profile then misses by construction.")
    for pr in PROFILES:
        print(f"  {pr}  {'meets' if pr.latency_s <= d else 'MISSES'}")
    print()

    calibrate(args.safety_gb)
    if args.calibrate_only:
        return

    records = []
    for pressure in PRESSURES:
        blk = hold(pressure)
        for policy in ("static", "switching"):
            out = args.outdir / f"p{pressure:g}_{policy}_{args.placement}.json"
            cmd = [PY, WORKER, "--policy", policy, "--placement", args.placement,
                   "--reps", str(args.reps), "--n-calls", str(args.n_calls),
                   "--clips", ",".join(CLIPS), "--pressure-gb", str(pressure),
                   "--output", str(out), "--safety-gb", str(args.safety_gb)]
            r = subprocess.run(cmd, capture_output=True, text=True)
            for line in r.stdout.splitlines():
                if line.startswith("["):
                    print("  " + line)
            if out.exists():
                records.append(json.loads(out.read_text()))
            else:
                print(f"  [{policy} {pressure}GB] no output: {r.stderr.strip()[-300:]}")
        del blk
        torch.cuda.empty_cache()
        time.sleep(2.0)

    args.output.write_text(json.dumps(records, indent=2))
    print(f"\nSaved -> {args.output}")

    print("\n=== Summary ===")
    print(f"{'pressure':<11}{'policy':<12}{'profile':<10}{'completion':<13}"
          f"{'miss rate':<12}{'mean latency'}")
    for pressure in PRESSURES:
        for policy in ("static", "switching"):
            rs = [r for r in records
                  if r["pressure_gb"] == pressure and r["policy"] == policy]
            if not rs:
                continue
            reps = [x for r in rs for x in r["reps"]]
            comp = sum(x["completion_rate"] for x in reps) / len(reps)
            miss = sum(x["miss_rate"] for x in reps) / len(reps)
            lats = [l for x in reps for l in x["latencies"]]
            mean = sum(lats) / len(lats) if lats else float("nan")
            prof = rs[0]["profile_chosen"]["name"] if rs[0]["profile_chosen"] else "none"
            print(f"{pressure:<11.1f}{policy:<12}{prof:<10}{comp:<13.2f}"
                  f"{miss:<12.2f}{mean:.2f}")


if __name__ == "__main__":
    main()
