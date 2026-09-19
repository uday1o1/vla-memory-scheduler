"""Does upgrading residency pay for itself when memory frees up?

Holds memory so the model process starts on a small profile, then releases it
part way through the run. The adaptive arm should notice and upgrade; the fixed
arm stays small. Whether the upgrade is worth its cost is the question, and the
placement rule determines how many layers the upgrade has to move, so both
rules are run.

Four arms per clip:
  fixed    nested     never upgrades (placement is irrelevant, run as control)
  adaptive nested     upgrades, moving the minimum number of layers
  adaptive upstream   upgrades, moving however many the upstream rule requires

Release timing is driven by a marker the worker writes when it begins timed
calls, rather than by a fixed sleep, so it does not drift with model load time.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import torch

from pathlib import Path as _Path  # noqa: E402
sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))

from scheduler.paths import RESULTS, bootstrap  # noqa: E402

bootstrap()
from scheduler.profiles import deadline_s

WORKER = str(Path(__file__).resolve().parent / "worker.py")
PY = "/venv/main/bin/python"
MARKER = RESULTS / "dyn_started.marker"
CLIPS = [
    "d497f01b-4f68-4c27-9a6c-55872a1d6bd6",
    "441057af-5c65-4d8e-993d-713090072248",
    "59aba96d-8920-4e49-8edb-c08bd800edf1",
    "8825b1fa-0abf-4a28-b444-25fef71cadcb",
]
PRESSURE_GB = 10.0     # forces the compact profile at startup
RELEASE_AFTER_S = 180  # about ten calls at the compact profile


def hold(gb: float):
    t = torch.empty(int(gb * (1024 ** 3) / 4), dtype=torch.float32, device="cuda:0")
    t.fill_(1.0)
    torch.cuda.synchronize()
    return t


def run_arm(policy: str, placement: str, clip: str, n_calls: int, outdir: Path):
    out = outdir / f"{policy}_{placement}_{clip[:8]}.json"
    MARKER.unlink(missing_ok=True)
    blk = hold(PRESSURE_GB)

    cmd = [PY, WORKER, "--policy", policy, "--placement", placement,
           "--n-calls", str(n_calls), "--clip-id", clip,
           "--output", str(out), "--marker", str(MARKER)]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    # Wait for the worker to reach its timed calls before starting the clock,
    # so release timing does not drift with how long the model took to load.
    waited = 0.0
    while not MARKER.exists() and proc.poll() is None and waited < 600:
        time.sleep(1.0)
        waited += 1.0

    if proc.poll() is None:
        time.sleep(RELEASE_AFTER_S)
        del blk
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        print(f"    released {PRESSURE_GB}GB")
    else:
        del blk
        torch.cuda.empty_cache()

    stdout, _ = proc.communicate()
    for line in stdout.decode().splitlines():
        if line.startswith("["):
            print("    " + line)
    return json.loads(out.read_text()) if out.exists() else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-calls", type=int, default=30)
    p.add_argument("--outdir", type=Path, default=RESULTS / "dyn")
    p.add_argument("--output", type=Path, default=RESULTS / "dynamic_results.json")
    args = p.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    torch.cuda.set_device(0)

    print(f"Deadline {deadline_s():.3f}s. Start under {PRESSURE_GB}GB pressure, "
          f"release after ~{RELEASE_AFTER_S}s of timed calls.\n")

    arms = [("fixed", "nested"), ("adaptive", "nested"), ("adaptive", "upstream")]
    records = []
    for clip in CLIPS:
        print(f"clip {clip[:8]}")
        for policy, placement in arms:
            r = run_arm(policy, placement, clip, args.n_calls, args.outdir)
            if r:
                r["clip_id"] = clip
                records.append(r)
        print()

    args.output.write_text(json.dumps(records, indent=2))
    print(f"Saved -> {args.output}\n")

    print("=== Summary ===")
    print(f"{'policy':<10}{'placement':<11}{'switches':<10}{'transition s':<15}"
          f"{'mean inference':<16}{'misses'}")
    for policy, placement in arms:
        rs = [r for r in records
              if r["policy"] == policy and r["placement"] == placement]
        if not rs:
            continue
        calls = [c for r in rs for c in r["calls"]]
        if not calls:
            continue
        sw = sum(1 for c in calls if c["switched_from"])
        tr = sum(c["transition_s"] for c in calls)
        inf = sum(c["inference_s"] for c in calls) / len(calls)
        miss = sum(1 for c in calls if not c["met_deadline"])
        print(f"{policy:<10}{placement:<11}{sw:<10}{tr:<15.2f}{inf:<16.2f}"
              f"{miss}/{len(calls)}")

    print("\nThe question is whether upgrading pays for itself. Compare the fixed")
    print("arm's miss count against the adaptive arms, and compare the two")
    print("placement rules against each other on transition time.")


if __name__ == "__main__":
    main()
