"""Does the residue scale with newly pinned layers, rather than with shedding?

A residency change was found to leave inference slower at the level it arrives
at, but only when shedding layers, not when restoring them. Reading the
upstream streaming path explains why that asymmetry is not about shedding as
such.

Non-resident layers are held in pinned host memory, because the prefetch
submits `copy_(..., non_blocking=True)` and that is asynchronous only from a
pinned source. The pinning routine pins any non-resident layer that is not
already pinned. A lower residency level has more non-resident layers, so
moving down requires fresh pinned allocations while moving up requires none:
the set needed after restoring is a subset of the set already pinned. Shedding
and pinning are therefore confounded in the earlier measurement, and shedding
is the wrong name for the cause.

The prediction that separates them is quantitative. If fresh pinned
allocations are what cost, the residue should grow with how many layers must
be newly pinned to reach the target, and a transition that needs none should
cost nothing however far it moves. Arriving at one target from several
starting levels varies the number of new pins while holding the target, the
clip and the measurement fixed.

A flat result refutes the pinning account and returns the cause to shedding
itself.
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
    p.add_argument("--target", type=int, default=16)
    p.add_argument("--from-levels", type=int, nargs="+", default=[17, 20, 24, 30])
    p.add_argument("--reps", type=int, default=6)
    p.add_argument("--settle", type=int, default=3)
    p.add_argument("--output", type=Path, default=RESULTS / "pinning_scaling.json")
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

    def pinned_count() -> int:
        """Layers whose parameters currently sit in pinned host memory."""
        n = 0
        for ly in loaded.vlm_layers:
            try:
                q = next(ly.parameters())
            except StopIteration:
                continue
            if q.device.type == "cpu" and q.numel() and q.is_pinned():
                n += 1
        return n

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

    print(f"\n=== baseline: straight to K={args.target} ===")
    pipe = place(args.target)
    base_pins = pinned_count()
    base = statistics.median(measure(pipe))
    print(f"  median {base:.3f}s   pinned layers {base_pins}")
    results["arms"]["baseline"] = {"from": None, "median": base,
                                   "pinned_after": base_pins, "new_pins": 0}
    pipe.remove(); del pipe; gc.collect()

    print(f"\n{'from K':<9}{'pins before':<14}{'pins after':<13}{'new pins':<11}"
          f"{'median s':<12}{'vs baseline'}")
    for start in args.from_levels:
        p0 = place(start)
        run_once(p0)
        before = pinned_count()
        p0.remove(); del p0; gc.collect()

        p1 = place(args.target)
        after = pinned_count()
        # Layers pinned to serve the target that were not pinned to serve the
        # start: the fresh allocations this transition required.
        new_pins = max(0, after - before)
        med = statistics.median(measure(p1))
        pct = 100 * (med - base) / base
        results["arms"][f"from_{start}"] = {
            "from": start, "median": med, "pinned_before": before,
            "pinned_after": after, "new_pins": new_pins, "pct_vs_baseline": pct,
        }
        print(f"{start:<9}{before:<14}{after:<13}{new_pins:<11}{med:<12.3f}{pct:+.2f}%")
        p1.remove(); del p1; gc.collect()

    # A repeat of the largest move, which should need no new pins the second
    # time if pinning is what costs.
    big = max(args.from_levels)
    print(f"\n=== repeat K={big} -> K={args.target}, second time ===")
    p0 = place(big); run_once(p0); before = pinned_count()
    p0.remove(); del p0; gc.collect()
    p1 = place(args.target); after = pinned_count()
    med = statistics.median(measure(p1))
    print(f"  pins {before} -> {after} (new {max(0, after - before)}), "
          f"median {med:.3f}s, {100 * (med - base) / base:+.2f}% vs baseline")
    results["arms"]["repeat"] = {"from": big, "median": med,
                                 "pinned_before": before, "pinned_after": after,
                                 "new_pins": max(0, after - before),
                                 "pct_vs_baseline": 100 * (med - base) / base}
    p1.remove(); del p1; gc.collect()

    args.output.write_text(json.dumps(results, indent=2))

    pts = [(v["new_pins"], v["pct_vs_baseline"])
           for k, v in results["arms"].items() if k.startswith("from_")]
    print("\n=== New pins against slowdown ===")
    for n, pct in sorted(pts):
        print(f"  {n:>3} new pins -> {pct:+.2f}%")
    if len(pts) >= 2 and max(n for n, _ in pts) > min(n for n, _ in pts):
        lo = min(pts, key=lambda x: x[0]); hi = max(pts, key=lambda x: x[0])
        print(f"\n  {lo[0]} pins costs {lo[1]:+.2f}% and {hi[0]} pins costs {hi[1]:+.2f}%.")
        if hi[1] > lo[1] + 1:
            print("  The cost grows with fresh pinned allocations, so pinning is the")
            print("  mechanism and shedding is only the thing that forces it.")
        else:
            print("  The cost does not grow with fresh pinned allocations, so pinning")
            print("  is not the mechanism and the cause lies elsewhere.")
    print(f"\nSaved -> {args.output}")


if __name__ == "__main__":
    main()
