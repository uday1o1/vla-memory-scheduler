"""Why is a residency level reached by transition slower than one started at?

On one machine, inference at K=16 runs about six percent slower if that level
was reached by changing residency than if the process started there, and the
gap does not decay. The observation is solid and the cause is not known. Until
it is, it is a curiosity rather than a result, because nothing follows from it
about how to build the system.

Three questions decide it, and each is a comparison against the same baseline:
inference at the target level in a process that never changed residency.

  Direction. Shedding layers pins host memory and costs three to six times
  what restoring them does. If only the shedding direction leaves the residue,
  the pinned-memory path is implicated rather than anything generic.

  The allocator. If emptying the caching allocator after the change restores
  the baseline, the residue is cached blocks left in a state that later
  allocations cannot use, which is a known failure mode and one a system can
  act on.

  Evidence in the allocator's own counters. Retried allocations and inactive
  split bytes are what fragmentation looks like from inside. If they rise
  across a transition and fall when the baseline is restored, that is direct
  support rather than inference from timing alone.

A mitigation that works matters as much as the mechanism: an adaptive policy
that must pay six percent forever after each change is a different proposition
from one that pays it until it calls a function.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import statistics
import sys
import time
from pathlib import Path
from pathlib import Path as _Path

sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch  # noqa: E402

from scheduler.paths import R1_CONFIG, RESULTS, bootstrap  # noqa: E402

bootstrap()

from alpamayo_memopt import load_config  # noqa: E402
from alpamayo_memopt.models import TriHookPipeline, get_adapter  # noqa: E402
from scheduler.inputs import prepare_inputs_for_clip  # noqa: E402
from scheduler.placement import nested_placement  # noqa: E402
from scheduler.provenance import run_metadata  # noqa: E402

CLIP = "d497f01b-4f68-4c27-9a6c-55872a1d6bd6"
GB = 1024 ** 3


def alloc_stats() -> dict:
    """The allocator's own view, which is where fragmentation is visible."""
    s = torch.cuda.memory_stats()
    return {
        "allocated_gb": s.get("allocated_bytes.all.current", 0) / GB,
        "reserved_gb": s.get("reserved_bytes.all.current", 0) / GB,
        "inactive_split_gb": s.get("inactive_split_bytes.all.current", 0) / GB,
        "num_alloc_retries": s.get("num_alloc_retries", 0),
        "num_ooms": s.get("num_ooms", 0),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--target", type=int, default=16, help="level every arm is measured at")
    p.add_argument("--from-above", type=int, default=24, help="level shed down from")
    p.add_argument("--from-below", type=int, default=8, help="level restored up from")
    p.add_argument("--reps", type=int, default=8)
    p.add_argument("--settle", type=int, default=3)
    p.add_argument("--output", type=Path, default=RESULTS / "transition_residue.json")
    args = p.parse_args()

    from transformers.utils import logging as _hf
    _hf.set_verbosity_error(); _hf.disable_progress_bar()
    device = "cuda:0"
    torch.cuda.set_device(0)

    config = load_config(R1_CONFIG)
    adapter = get_adapter(config.model.kind)

    class A: pass
    a = A()
    a.model_id = None; a.model_cache_dir = None; a.model_revision = None
    a.attn_implementation = None; a.local_files_only = False
    a.num_traj_samples = adapter.default_num_traj_samples
    a.max_generation_length = adapter.default_max_generation_length

    print("Loading model...")
    loaded = adapter.load(a, config)
    adapter.setup_essentials(loaded, device)
    n_vlm = len(loaded.vlm_layers)
    inputs = prepare_inputs_for_clip(loaded, CLIP, device)
    current: set[int] = set()

    def place(k: int):
        nonlocal current
        t = set(nested_placement(k, n_vlm))
        for i in t - current:
            loaded.vlm_layers[i].to(device)
        for i in current - t:
            loaded.vlm_layers[i].to("cpu")
        torch.cuda.synchronize()
        current = t
        return TriHookPipeline(loaded.vlm_layers, loaded.vit_blocks,
                               loaded.expert_layers, list(t), device=device)

    def run_once(pipe) -> float:
        t0 = time.perf_counter()
        pipe.start_iteration()
        with torch.no_grad():
            adapter.run(loaded, inputs, a)
        torch.cuda.synchronize()
        return time.perf_counter() - t0

    def measure(pipe) -> list[float]:
        for _ in range(args.settle):
            run_once(pipe)
        return [run_once(pipe) for _ in range(args.reps)]

    results = {"meta": run_metadata(args), "target": args.target, "arms": {}}

    def record(name: str, times: list[float], note: str = "") -> float:
        med = statistics.median(times)
        results["arms"][name] = {"times": times, "median": med,
                                 "alloc": alloc_stats(), "note": note}
        return med

    # Baseline: the target level in a process that has never changed residency.
    print(f"\n=== baseline: straight to K={args.target}, no prior change ===")
    pipe = place(args.target)
    base = record("baseline", measure(pipe))
    print(f"  median {base:.3f}s   {alloc_stats()}")
    pipe.remove(); del pipe; gc.collect()

    def arm(name: str, start_k: int, clear: bool) -> None:
        nonlocal base
        print(f"\n=== {name}: K={start_k} -> K={args.target}"
              f"{', then empty_cache' if clear else ''} ===")
        p0 = place(start_k)
        run_once(p0)
        p0.remove(); del p0; gc.collect()

        p1 = place(args.target)
        if clear:
            torch.cuda.empty_cache()
        med = record(name, measure(p1))
        print(f"  median {med:.3f}s   {med / base * 100 - 100:+.2f}% vs baseline")
        print(f"  {alloc_stats()}")
        p1.remove(); del p1; gc.collect()

    arm("shed", args.from_above, clear=False)
    arm("shed_then_empty_cache", args.from_above, clear=True)
    arm("restore", args.from_below, clear=False)
    arm("restore_then_empty_cache", args.from_below, clear=True)

    args.output.write_text(json.dumps(results, indent=2))

    print(f"\n=== Against the baseline of {base:.3f}s ===")
    print(f"{'arm':<28}{'median s':<12}{'vs baseline':<14}{'retries':<10}{'inactive split GB'}")
    for name, blk in results["arms"].items():
        al = blk["alloc"]
        print(f"{name:<28}{blk['median']:<12.3f}"
              f"{blk['median'] / base * 100 - 100:<+14.2f}"
              f"{al['num_alloc_retries']:<10}{al['inactive_split_gb']:.3f}")

    shed = results["arms"]["shed"]["median"]
    shed_c = results["arms"]["shed_then_empty_cache"]["median"]
    rest = results["arms"]["restore"]["median"]
    print()
    if shed / base > 1.02 and rest / base <= 1.02:
        print("  Only shedding leaves the residue, so the pinned host memory that")
        print("  offloading allocates is implicated rather than generic churn.")
    elif shed / base > 1.02 and rest / base > 1.02:
        print("  Both directions leave it, so it is not specific to offloading.")
    else:
        print("  Neither direction reproduced the residue on this machine.")
    if shed / base > 1.02:
        print(f"  Emptying the cache recovers "
              f"{100 * (shed - shed_c) / (shed - base):.0f}% of the loss, so it is "
              f"{'a usable mitigation' if shed_c / base < 1.01 else 'not a full fix'}.")
    print(f"\nSaved -> {args.output}")


if __name__ == "__main__":
    main()
