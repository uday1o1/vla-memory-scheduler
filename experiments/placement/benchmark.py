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
import gc
import json
import os
import statistics
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch

from pathlib import Path as _Path  # noqa: E402
sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))

from ias.paths import R1_CONFIG, RESULTS, bootstrap  # noqa: E402

bootstrap()
from alpamayo_memopt import load_config
from alpamayo_memopt.models import TriHookPipeline, get_adapter
from alpamayo_memopt.profiler import interleaved_placement
from ias.inputs import prepare_inputs_for_clip
from ias.placement import nested_placement
from ias.provenance import run_metadata

CLIPS = [
    "d497f01b-4f68-4c27-9a6c-55872a1d6bd6",
    "441057af-5c65-4d8e-993d-713090072248",
    "59aba96d-8920-4e49-8edb-c08bd800edf1",
    "8825b1fa-0abf-4a28-b444-25fef71cadcb",
    "89e4e404-80ea-42e2-bf8e-8395af5ab111",
    "7d109673-d967-4b02-93c5-6d2d25d964d0",
]

def build_paths(max_k: int) -> dict[int, list[int]]:
    """Adaptation paths of varying step size, scaled to what the card can hold.

    Each path oscillates around a centre so total displacement stays
    comparable while the step size varies. Centring on max_k - 3 keeps every
    excursion inside the card's capacity; hardcoding counts sized for a 24GB
    card runs out of memory on a 16GB one.
    """
    c = max_k - 3
    paths = {}
    for step in (1, 2, 3, 5, 8, 12):
        seq, k, direction = [c], c, 1
        for i in range(8):
            if i % 4 == 3:
                direction *= -1
            k = max(2, min(max_k, k + direction * step))
            seq.append(k)
        paths[step] = seq
    return paths


def build_latency_ks(max_k: int) -> list[int]:
    """Residency levels for the latency-parity check, spread over the range."""
    return sorted({max(2, int(max_k * f)) for f in (0.5, 0.75, 1.0)})


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--transition-reps", type=int, default=5)
    p.add_argument("--latency-reps", type=int, default=3)
    p.add_argument("--max-k", type=int, default=33,
                   help="largest residency this card can hold; paths scale to it")
    p.add_argument("--output", type=Path, default=RESULTS / "placement_bench.json")
    args = p.parse_args()

    from transformers.utils import logging as _hf
    _hf.set_verbosity_error(); _hf.disable_progress_bar()
    PATHS = build_paths(args.max_k)
    LATENCY_KS = build_latency_ks(args.max_k)
    print(f"max_k={args.max_k}  latency Ks={LATENCY_KS}")

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

    results = {"meta": run_metadata(args), "transition": {}, "latency": {}}

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
                    # Each TriHookPipeline holds two device buffers sized for
                    # the largest streamed layer. Rebinding `pipe` does not
                    # free the previous one promptly, so across dozens of
                    # transitions the stale buffers accumulate into several GB
                    # and exhaust a 16GB card. Dropping it explicitly before
                    # the timer keeps memory bounded without charging the cost
                    # to the measurement.
                    #
                    # empty_cache() is deliberately NOT called here. Returning
                    # the blocks to the allocator's pool is what a real
                    # adaptive system would have; forcing a full release would
                    # make every transition pay a fresh cudaMalloc and measure
                    # a pessimistic case that does not occur in practice.
                    del pipe
                    gc.collect()
                    t0 = time.perf_counter()
                    pipe = switch(sorted(tgt))
                    total += time.perf_counter() - t0
                pipe.remove()
                times.append(total)
                moves_total = mv
                # Release the allocator's cached blocks between repetitions.
                # Each transition leaves freed blocks cached, and across dozens
                # of transitions that accumulates until a 16GB card runs out,
                # even at residency levels that fit comfortably on their own.
                # This sits outside the timed region deliberately: calling it
                # inside switch() would add its cost to the transition times
                # this benchmark exists to measure.
                torch.cuda.empty_cache()
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
