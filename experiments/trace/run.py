"""Static residency against profile switching, under a tenant whose memory
demand changes, with deadline-miss rate as the measure.

This is the comparison the project was asked to make, run where it can
discriminate. The driver plays a co-located tenant: before each call of the
driving model it gives the tenant its demand from the trace, by allocating
real device memory in half-gigabyte chunks, and the allocation succeeding is
what admits it. The driving model runs in a separate worker process and
responds according to its policy (scheduler/trace.py).

Arms, each a fresh process, in an order shuffled per repetition:
  static_fast, static_compact, startup      never change residency
  runtime x {nested, upstream} placement x {copy, clean} demotion
  deadline x {nested, upstream} placement, copy demotion

The four runtime arms are a two-by-two ablation of the two ways to make a
residency change cheaper. The deadline arms test a third: changing less. Static arms never change residency, so their
latency does not depend on the churn rate and they run at the first one
only unless --static-every-churn is given.

Before running, check that the design can separate the arms:
    python experiments/trace/simulate.py
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch  # noqa: E402

from scheduler.paths import RESULTS  # noqa: E402
from scheduler.provenance import run_metadata  # noqa: E402
from scheduler.trace import check_design, expand  # noqa: E402

WORKER = str(Path(__file__).resolve().parent / "worker.py")
CLIPS = [
    "d497f01b-4f68-4c27-9a6c-55872a1d6bd6",
    "441057af-5c65-4d8e-993d-713090072248",
    "59aba96d-8920-4e49-8edb-c08bd800edf1",
    "8825b1fa-0abf-4a28-b444-25fef71cadcb",
    "89e4e404-80ea-42e2-bf8e-8395af5ab111",
    "7d109673-d967-4b02-93c5-6d2d25d964d0",
]
STATIC = [("static_fast", "nested", "copy"), ("static_compact", "nested", "copy"),
          ("startup", "nested", "copy")]
RUNTIME = [("runtime", pl, de) for pl in ("nested", "upstream") for de in ("copy", "clean")]
# The deadline policy under the upstream demotion mechanism only, to ask
# whether changing less often is by itself enough to avoid transition misses.
RUNTIME += [("deadline", pl, "copy") for pl in ("nested", "upstream")]
CHUNK_GB = 0.5


class Tenant:
    """Device memory held in chunks, so it can grow and shrink in steps."""

    def __init__(self):
        self.chunks: list[torch.Tensor] = []

    @property
    def held_gb(self) -> float:
        return len(self.chunks) * CHUNK_GB

    def shrink_to(self, gb: float) -> None:
        while self.held_gb > gb + 1e-9:
            self.chunks.pop()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    def grow_to(self, gb: float) -> bool:
        """All or nothing: a tenant that cannot get its whole demand keeps what
        it had, rather than holding a partial allocation that squeezes the
        driving model."""
        start = len(self.chunks)
        while self.held_gb < gb - 1e-9:
            try:
                c = torch.empty(int(CHUNK_GB * 1024 ** 3), dtype=torch.uint8, device="cuda:0")
                c.fill_(1)
                self.chunks.append(c)
            except torch.cuda.OutOfMemoryError:
                del self.chunks[start:]
                torch.cuda.empty_cache()
                return False
        torch.cuda.synchronize()
        return True


def wait_for(path: Path, proc: subprocess.Popen, timeout_s: float = 900) -> bool:
    t0 = time.time()
    while not path.exists():
        if proc.poll() is not None or time.time() - t0 > timeout_s:
            return False
        time.sleep(0.02)
    return True


def run_arm(arm, phase_calls, clip, outdir: Path) -> dict | None:
    policy, placement, demotion = arm
    name = f"{policy}_{placement}_{demotion}_L{phase_calls}_{clip[:8]}"
    sync = outdir / "sync" / name
    shutil.rmtree(sync, ignore_errors=True)
    sync.mkdir(parents=True)
    out = outdir / f"{name}.json"
    cmd = [sys.executable, WORKER, "--policy", policy, "--placement", placement,
           "--demotion", demotion, "--phase-calls", str(phase_calls),
           "--clip-id", clip, "--sync-dir", str(sync), "--output", str(out)]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    tenant = Tenant()
    if wait_for(sync / "loaded", proc):
        for i, demand in enumerate(expand(phase_calls)):
            if not wait_for(sync / f"pre{i}", proc):
                break
            t_pre = time.perf_counter()
            # Release first, so a worker that is about to grow can.
            tenant.shrink_to(min(tenant.held_gb, demand))
            (sync / f"ok{i}").write_text("")
            if not wait_for(sync / f"post{i}", proc):
                break
            admitted = tenant.grow_to(demand)
            # Written whole and then renamed into place: the worker starts
            # reading the moment the name exists, and would otherwise read an
            # empty file.
            tmp = sync / f".go{i}"
            tmp.write_text(json.dumps({
                "held_gb": tenant.held_gb, "admitted": admitted,
                "wait_s": time.perf_counter() - t_pre}))
            tmp.replace(sync / f"go{i}")
    stdout, _ = proc.communicate()
    tenant.shrink_to(0)
    shutil.rmtree(sync, ignore_errors=True)
    lines = stdout.decode(errors="replace").splitlines()
    summary = [ln for ln in lines if ln.startswith("[")]
    print("    " + (summary[-1] if summary else "worker produced no summary; tail: "
                    + " | ".join(lines[-3:])))
    return json.loads(out.read_text()) if out.exists() else None


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--reps", type=int, default=8)
    p.add_argument("--phase-calls", type=int, nargs="+", default=[6, 2])
    p.add_argument("--static-every-churn", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--outdir", type=Path, default=RESULTS / "trace")
    p.add_argument("--output", type=Path, default=RESULTS / "trace_results.json")
    p.add_argument("--rerun-failed", action="store_true",
                   help="rerun only the runs in --output that ended in an error, "
                        "each with its own arm, churn rate and clip, in place")
    args = p.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    torch.cuda.set_device(0)
    torch.empty(1, device="cuda:0")  # the tenant's context exists from the start
    rng = random.Random(args.seed)

    total_gb = torch.cuda.mem_get_info(0)[1] / 1024 ** 3
    for L in args.phase_calls:
        problems = check_design(total_gb, 1.0, L)
        if problems:
            sys.exit(f"The trace cannot discriminate on this card ({total_gb:.1f}GB) "
                     f"with these profiles: {problems}. Adjust PATTERN in "
                     f"scheduler/trace.py.")

    if args.rerun_failed:
        results = json.loads(args.output.read_text())
        for j, old in enumerate(results["runs"]):
            if not old.get("error"):
                continue
            a = old["args"]
            arm = (a["policy"], a["placement"], a["demotion"])
            print(f"rerun rep {old['rep'] + 1}, {arm}, L={a['phase_calls']}: {old['error']}")
            r = run_arm(arm, int(a["phase_calls"]), old["clip_id"], args.outdir)
            if r is not None:
                r.update(rep=old["rep"], clip_id=old["clip_id"], position=old["position"],
                         rerun_of_error=old["error"])
                results["runs"][j] = r
                args.output.write_text(json.dumps(results, indent=1))
        print(f"Saved -> {args.output}")
        return

    results = {"meta": run_metadata(args), "runs": []}
    for rep in range(args.reps):
        clip = CLIPS[rep % len(CLIPS)]
        jobs = [(arm, L) for L in args.phase_calls for arm in RUNTIME]
        jobs += [(arm, L) for L in (args.phase_calls if args.static_every_churn
                                   else args.phase_calls[:1]) for arm in STATIC]
        rng.shuffle(jobs)
        print(f"rep {rep + 1}/{args.reps}, clip {clip[:8]}")
        for position, (arm, L) in enumerate(jobs):
            r = run_arm(arm, L, clip, args.outdir)
            if r is not None:
                r.update(rep=rep, clip_id=clip, position=position)
                results["runs"].append(r)
            # Saved after every arm, so an interrupted run keeps what it has.
            args.output.write_text(json.dumps(results, indent=1))
    print(f"Saved -> {args.output}")


if __name__ == "__main__":
    main()
