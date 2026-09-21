"""How long does a residency change keep costing after it completes?

The direct cost of changing residency is the time to move layers, which is
measured elsewhere. This measures what happens afterwards. A block of twelve
residency changes was found to leave the next sustained inference about
thirteen percent slower regardless of which placement rule was in use, so the
change is still being paid for after the layers have stopped moving.

That matters for the question of whether adaptivity pays for itself. An
adaptive policy generates residency changes by design, and if each one is
followed by a recovery period, the true cost of a change is the migration plus
the recovery, not the migration alone.

The measurement records latency by position after a change, against a control
that performs no change at all, so the recovery can be read as a curve rather
than assumed to be a single slow call. Summing the excess over the control
across positions gives the recovery cost of one change, which adds to the
migration cost already measured.
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


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--k-a", type=int, default=24, help="residency level switched from")
    p.add_argument("--k-b", type=int, default=16, help="residency level switched to")
    p.add_argument("--tail", type=int, default=14,
                   help="inference calls recorded after each change")
    p.add_argument("--cycles", type=int, default=4,
                   help="how many changes to average over")
    p.add_argument("--output", type=Path, default=RESULTS / "churn_recovery.json")
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

    def switch(indices):
        nonlocal current
        t = set(indices)
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

    results = {"meta": run_metadata(args), "k_a": args.k_a, "k_b": args.k_b}

    # Control: no residency change at all. Settle first, then record the same
    # number of calls, so any drift unrelated to switching is visible.
    print(f"\n=== Control: no residency change, steady at K={args.k_b} ===")
    pipe = switch(nested_placement(args.k_b, n_vlm))
    for _ in range(8):
        run_once(pipe)
    control = [run_once(pipe) for _ in range(args.tail)]
    pipe.remove(); del pipe; gc.collect()
    steady = statistics.median(control)
    print(f"  median {steady:.3f}s   range {min(control):.3f} to {max(control):.3f}s")

    # Treatment: change residency, then record the tail of calls that follow.
    print(f"\n=== After a change from K={args.k_a} to K={args.k_b} ===")
    by_position: list[list[float]] = [[] for _ in range(args.tail)]
    for c in range(args.cycles):
        pipe = switch(nested_placement(args.k_a, n_vlm))
        for _ in range(4):
            run_once(pipe)
        pipe.remove(); del pipe; gc.collect()

        pipe = switch(nested_placement(args.k_b, n_vlm))
        for i in range(args.tail):
            by_position[i].append(run_once(pipe))
        pipe.remove(); del pipe; gc.collect()
        print(f"  cycle {c + 1}/{args.cycles} done")

    print(f"\n{'call':<7}{'median s':<12}{'vs steady':<12}{'excess s'}")
    curve, excess_total = [], 0.0
    for i, vals in enumerate(by_position, start=1):
        med = statistics.median(vals)
        pct = 100 * (med - steady) / steady
        excess = med - steady
        if excess > 0:
            excess_total += excess
        curve.append({"call": i, "median": med, "pct_vs_steady": pct,
                      "values": vals})
        print(f"{i:<7}{med:<12.3f}{pct:<+12.2f}%{excess:+.3f}")

    results["control"] = {"values": control, "median": steady}
    results["after_change"] = curve
    results["recovery_excess_s"] = excess_total

    elevated = [c["call"] for c in curve if c["pct_vs_steady"] > 2]
    print(f"\n=== Recovery ===")
    print(f"  steady-state median        {steady:.3f}s")
    print(f"  calls above steady by 2%   {elevated if elevated else 'none'}")
    print(f"  total excess over the tail {excess_total:.2f}s")
    print("\n  This excess is paid after the layers have finished moving, so it")
    print("  adds to the migration cost rather than being part of it. A policy")
    print("  that changes residency often pays it every time.")

    args.output.write_text(json.dumps(results, indent=2))
    print(f"\nSaved -> {args.output}")


if __name__ == "__main__":
    main()
