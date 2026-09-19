"""The compute-contention boundary, retested against the proposed policy.

The earlier compute-contention result was collected with a continuous-K policy
that has since been withdrawn. A boundary claim should be tested against the
design actually being proposed, so this repeats it with profile switching.

The expectation, stated before running: compute contention does not change how
much GPU memory is free, so the switching policy will select the same profile
as the static policy and behave identically. If that holds, it establishes the
boundary cleanly, that residency selection is a memory-domain lever and has no
purchase on compute interference. If switching somehow differs, the reasoning
about what drives the policy is wrong and needs revisiting.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from pathlib import Path as _Path  # noqa: E402
sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))

from scheduler.paths import REPO_ROOT, RESULTS, bootstrap  # noqa: E402

bootstrap()
from scheduler.profiles import deadline_s

WORKER = str(REPO_ROOT / "experiments" / "switching" / "worker.py")
CONTENTION = str(REPO_ROOT / "scheduler" / "contention.py")
PY = "/venv/main/bin/python"
CLIPS = [
    "d497f01b-4f68-4c27-9a6c-55872a1d6bd6",
    "441057af-5c65-4d8e-993d-713090072248",
    "59aba96d-8920-4e49-8edb-c08bd800edf1",
    "8825b1fa-0abf-4a28-b444-25fef71cadcb",
    "89e4e404-80ea-42e2-bf8e-8395af5ab111",
    "7d109673-d967-4b02-93c5-6d2d25d964d0",
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--reps", type=int, default=8)
    p.add_argument("--n-calls", type=int, default=20)
    p.add_argument("--duty-cycle", type=float, default=0.70)
    p.add_argument("--outdir", type=Path, default=RESULTS / "cbound")
    p.add_argument("--output", type=Path, default=RESULTS / "compute_boundary_results.json")
    args = p.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    print(f"Deadline {deadline_s():.3f}s. Steady compute contention at "
          f"{args.duty_cycle:.0%} duty cycle, independent process under CUDA MPS.\n")
    print("Expectation stated in advance: compute contention leaves free memory")
    print("unchanged, so switching should select the same profile as static and")
    print("behave identically, establishing the boundary.\n")

    records = []
    for policy in ("static", "switching"):
        out = args.outdir / f"compute_{policy}.json"

        # Contention runs as an independent process sharing the GPU through MPS,
        # started before the model process and killed after it finishes.
        cont = subprocess.Popen(
            [PY, CONTENTION, "--mode", "steady", "--duty-cycle", str(args.duty_cycle),
             "--duration", "36000"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(5.0)

        try:
            r = subprocess.run(
                [PY, WORKER, "--policy", policy, "--placement", "nested",
                 "--reps", str(args.reps), "--n-calls", str(args.n_calls),
                 "--clips", ",".join(CLIPS), "--pressure-gb", "0",
                 "--output", str(out)],
                capture_output=True, text=True)
            for line in r.stdout.splitlines():
                if line.startswith("["):
                    print("  " + line)
            if out.exists():
                rec = json.loads(out.read_text())
                rec["contention"] = f"steady {args.duty_cycle:.0%} duty cycle"
                records.append(rec)
            else:
                print(f"  [{policy}] no output: {r.stderr.strip()[-300:]}")
        finally:
            cont.terminate()
            cont.wait(timeout=10)
            time.sleep(3.0)

    args.output.write_text(json.dumps(records, indent=2))
    print(f"\nSaved -> {args.output}")

    if len(records) == 2:
        s, w = records[0], records[1]
        sp = s["profile_chosen"]["name"] if s["profile_chosen"] else "none"
        wp = w["profile_chosen"]["name"] if w["profile_chosen"] else "none"
        sm = sum(x["miss_rate"] for x in s["reps"]) / len(s["reps"])
        wm = sum(x["miss_rate"] for x in w["reps"]) / len(w["reps"])
        sl = [l for x in s["reps"] for l in x["latencies"]]
        wl = [l for x in w["reps"] for l in x["latencies"]]
        print("\n=== Boundary check ===")
        print(f"  static    profile={sp:<8} miss={sm:.2f} "
              f"mean={sum(sl)/len(sl) if sl else float('nan'):.2f}s")
        print(f"  switching profile={wp:<8} miss={wm:.2f} "
              f"mean={sum(wl)/len(wl) if wl else float('nan'):.2f}s")
        if sp == wp:
            print("\n  Same profile chosen. Residency selection has no purchase on")
            print("  compute interference, because compute contention does not change")
            print("  how much memory is free. The boundary holds.")
        else:
            print("\n  Profiles differ, which contradicts the stated expectation and")
            print("  means the reasoning about what drives the policy needs revisiting.")


if __name__ == "__main__":
    main()
