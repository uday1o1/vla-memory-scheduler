"""Comprehensive empirical validation of the nested placement rule.

Two claims need separate treatment because they have different dependencies:

  A. Transition cost. Depends only on layer movement, not on scene content,
     so it is measured across adaptation paths of varying granularity with
     repetitions for variance, not across clips.

  B. Latency parity. The claim that nested matches upstream's steady-state
     latency must hold across scene content, so it is measured across all
     six clips at several residency levels.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch

sys.path.insert(0, "/root/oom-free-alpamayo")
sys.path.insert(0, "/root")
from alpamayo_memopt import load_config
from alpamayo_memopt.models import TriHookPipeline, get_adapter
from alpamayo_memopt.profiler import interleaved_placement
from ias_calibrate import prepare_inputs_for_clip
from ias_placement import nested_placement

CLIPS = [
    "d497f01b-4f68-4c27-9a6c-55872a1d6bd6",
    "441057af-5c65-4d8e-993d-713090072248",
    "59aba96d-8920-4e49-8edb-c08bd800edf1",
    "8825b1fa-0abf-4a28-b444-25fef71cadcb",
    "89e4e404-80ea-42e2-bf8e-8395af5ab111",
    "7d109673-d967-4b02-93c5-6d2d25d964d0",
]

# Adaptation paths spanning a range of granularities. Each oscillates around
# a centre so total displacement is comparable while step size varies.
PATHS = {
    1: [30, 31, 30, 29, 30, 31, 32, 31, 30],
    2: [30, 28, 30, 32, 30, 28, 26, 28, 30],
    3: [30, 27, 30, 33, 30, 27, 24, 27, 30],
    5: [30, 25, 30, 33, 28, 23, 18, 23, 28],
    8: [30, 22, 30, 22, 30, 22, 30, 22, 30],
    12: [30, 18, 30, 18, 30, 18, 30, 18, 30],
}
LATENCY_KS = [16, 24, 33]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--transition-reps", type=int, default=5)
    p.add_argument("--latency-reps", type=int, default=3)
    p.add_argument("--output", type=Path, default=Path("/root/placement_bench.json"))
    args = p.parse_args()

    from transformers.utils import logging as _hf
    _hf.set_verbosity_error(); _hf.disable_progress_bar()
    device = "cuda:0"
    torch.cuda.set_device(0)

    config = load_config("/root/oom-free-alpamayo/r1_config.json")
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

    results = {"transition": {}, "latency": {}}

    # ---- A. transition cost versus adaptation granularity ----
    print(f"\n=== A. Transition cost by granularity ({args.transition_reps} reps) ===")
    print(f"{'mean|dK|':<10}{'upstream s':<22}{'nested s':<22}{'speedup':<10}{'moves u/n'}")
    for gran, path in sorted(PATHS.items()):
        per_rule = {}
        for name, rule in (("upstream", interleaved_placement), ("nested", nested_placement)):
            times, moves_total = [], 0
            for _ in range(args.transition_reps):
                pipe = switch(rule(path[0], n_vlm))
                total = 0.0
                mv = 0
                for k in path[1:]:
                    tgt = set(rule(k, n_vlm))
                    mv += len(tgt - current) + len(current - tgt)
                    pipe.remove()
                    t0 = time.perf_counter()
                    pipe = switch(sorted(tgt))
                    total += time.perf_counter() - t0
                pipe.remove()
                times.append(total)
                moves_total = mv
            per_rule[name] = {"times": times, "moves": moves_total,
                              "mean": statistics.mean(times),
                              "stdev": statistics.stdev(times) if len(times) > 1 else 0.0}
        u, n = per_rule["upstream"], per_rule["nested"]
        results["transition"][gran] = per_rule
        print(f"{gran:<10}{u['mean']:.3f} +/- {u['stdev']:<12.3f}"
              f"{n['mean']:.3f} +/- {n['stdev']:<12.3f}"
              f"{u['mean'] / n['mean']:<10.2f}{u['moves']}/{n['moves']}")

    # ---- B. latency parity across clips ----
    print(f"\n=== B. Steady-state latency parity across {len(CLIPS)} clips "
          f"({args.latency_reps} reps) ===")
    print(f"{'K':<6}{'clip':<10}{'upstream s':<14}{'nested s':<14}{'delta':<10}{'pct'}")
    for k in LATENCY_KS:
        for clip in CLIPS:
            inputs = prepare_inputs_for_clip(loaded, clip, device)
            row = {}
            for name, rule in (("upstream", interleaved_placement), ("nested", nested_placement)):
                pipe = switch(rule(k, n_vlm))
                pipe.start_iteration()
                with torch.no_grad():
                    adapter.run(loaded, inputs, a)
                torch.cuda.synchronize()
                ts = []
                for _ in range(args.latency_reps):
                    t0 = time.perf_counter()
                    pipe.start_iteration()
                    with torch.no_grad():
                        adapter.run(loaded, inputs, a)
                    torch.cuda.synchronize()
                    ts.append(time.perf_counter() - t0)
                pipe.remove()
                row[name] = statistics.mean(ts)
            results["latency"].setdefault(str(k), {})[clip] = row
            d = row["nested"] - row["upstream"]
            print(f"{k:<6}{clip[:8]:<10}{row['upstream']:<14.3f}{row['nested']:<14.3f}"
                  f"{d:<+10.3f}{100 * d / row['upstream']:+.2f}%")

    args.output.write_text(json.dumps(results, indent=2))

    print("\n=== Summary ===")
    deltas = [v["nested"] - v["upstream"]
              for ks in results["latency"].values() for v in ks.values()]
    pcts = [100 * (v["nested"] - v["upstream"]) / v["upstream"]
            for ks in results["latency"].values() for v in ks.values()]
    print(f"Latency delta across all {len(deltas)} clip/K pairs: "
          f"mean {statistics.mean(deltas):+.4f}s ({statistics.mean(pcts):+.3f}%), "
          f"max |delta| {max(abs(d) for d in deltas):.4f}s")
    print(f"Transition speedup by granularity: " + ", ".join(
        f"|dK|={g}: {v['upstream']['mean'] / v['nested']['mean']:.2f}x"
        for g, v in sorted(results["transition"].items())))
    print(f"\nSaved -> {args.output}")


if __name__ == "__main__":
    main()
