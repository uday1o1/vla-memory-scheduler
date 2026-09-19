"""Co-location capacity experiment.

Compares three configurations on one RTX 3090:
  1. one tenant at K=33  (what the static offline policy picks: 17.24GB)
  2. one tenant at K=16  (reduced residency, solo - isolates the residency cost)
  3. two tenants at K=16 (11.28GB each - what reduced residency enables)

Config 3 is only reachable if residency is chosen with available VRAM in
mind; the static K=33 choice forecloses it. Comparing 2 against 3 separates
the residency cost from the co-location contention cost.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

WORKER = str(Path(__file__).resolve().parent / "worker.py")
PY = "/venv/main/bin/python"
CLIPS = [
    "d497f01b-4f68-4c27-9a6c-55872a1d6bd6",
    "441057af-5c65-4d8e-993d-713090072248",
]


def run_config(name: str, ks: list[int], n_calls: int, outdir: Path) -> dict:
    """Launch one tenant per entry in `ks`, all starting their timed calls together."""
    print(f"\n=== {name}: {len(ks)} tenant(s) at K={ks} ===")
    # Model load takes ~1-2 min; give every tenant time to reach the barrier.
    start_at = time.time() + 240
    procs, outputs = [], []
    for idx, k in enumerate(ks):
        out = outdir / f"{name}_t{idx}.json"
        outputs.append(out)
        cmd = [PY, WORKER, "--k", str(k), "--n-calls", str(n_calls),
               "--start-at", str(start_at), "--output", str(out),
               "--clip-id", CLIPS[idx % len(CLIPS)], "--tenant-id", f"t{idx}"]
        procs.append(subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT))

    for p in procs:
        out, _ = p.communicate()
        for line in out.decode().splitlines():
            if line.startswith("[") or "Error" in line or "error" in line:
                print("   ", line)

    records = []
    for o in outputs:
        if o.exists():
            records.append(json.loads(o.read_text()))
        else:
            records.append({"error": "no output file", "latencies": [], "load_ok": False})

    served = sum(1 for r in records if r.get("load_ok") and r.get("latencies"))
    all_lat = [l for r in records for l in r.get("latencies", [])]
    mean = sum(all_lat) / len(all_lat) if all_lat else float("nan")
    print(f"    tenants served: {served}/{len(ks)}   mean per-call latency: {mean:.3f}s")
    return {"name": name, "ks": ks, "tenants_served": served, "records": records}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-calls", type=int, default=20)
    p.add_argument("--outdir", type=Path, default=RESULTS / "coloc")
    p.add_argument("--output", type=Path, default=RESULTS / "colocation_results.json")
    args = p.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    results = []
    results.append(run_config("solo_k33", [33], args.n_calls, args.outdir))
    results.append(run_config("solo_k16", [16], args.n_calls, args.outdir))
    results.append(run_config("dual_k16", [16, 16], args.n_calls, args.outdir))
    # Does the static choice actually foreclose co-location? Try it.
    results.append(run_config("dual_k33", [33, 33], args.n_calls, args.outdir))

    args.output.write_text(json.dumps(results, indent=2))
    print(f"\nSaved -> {args.output}")

    print("\n=== Summary ===")
    print(f"{'config':<12}{'tenants served':<18}{'mean latency':<16}{'throughput (calls/s)'}")
    for r in results:
        lat = [l for rec in r["records"] for l in rec.get("latencies", [])]
        if lat:
            mean = sum(lat) / len(lat)
            tput = r["tenants_served"] / mean
            print(f"{r['name']:<12}{r['tenants_served']:<18}{mean:<16.3f}{tput:.4f}")
        else:
            print(f"{r['name']:<12}{r['tenants_served']:<18}{'n/a':<16}{'n/a'}")


if __name__ == "__main__":
    main()
