"""Does layer spread affect inference latency at all?

The upstream rule spreads resident layers evenly, and the nested rule was
built to match that spread, on the assumption that a long run of consecutive
non-resident layers is expensive because the streaming pipeline must cover it.
That assumption has never been tested here. It matters for two reasons: if
spread does not drive latency, then the nested rule being worse on spread
costs nothing, and the upstream rule's even spacing is solving a problem that
does not exist.

The test varies spread by an order of magnitude while holding the residency
count fixed, so the resident layer count, the memory footprint and the amount
of streamed data are all identical and only the arrangement differs. A
sequential order is the extreme: at K=16 of 36 it leaves a run of 20
consecutive non-resident layers against 3 for the bisection rule.

If latency tracks spread, sequential should be markedly slower. If latency is
flat across that range, spread is not the mechanism and nestedness is the
whole of the placement story.
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
from alpamayo_memopt.profiler import interleaved_placement  # noqa: E402
from scheduler.inputs import prepare_inputs_for_clip  # noqa: E402
from scheduler.placement import max_gap, nested_placement  # noqa: E402
from scheduler.provenance import run_metadata  # noqa: E402

sys.path.insert(0, str(_Path(__file__).resolve().parent))
from baselines import order_greedy, order_sequential  # noqa: E402

CLIPS = [
    "d497f01b-4f68-4c27-9a6c-55872a1d6bd6",
    "441057af-5c65-4d8e-993d-713090072248",
    "59aba96d-8920-4e49-8edb-c08bd800edf1",
]


def rules_for(total: int):
    seq = order_sequential(total)
    greedy = order_greedy(total)
    return {
        "sequential": lambda k: sorted(seq[:k]),
        "bisection": lambda k: nested_placement(k, total),
        "greedy": lambda k: sorted(greedy[:k]),
        "upstream": lambda k: sorted(interleaved_placement(k, total)),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--k-values", type=int, nargs="+", default=[16, 24, 33])
    p.add_argument("--reps", type=int, default=5)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--output", type=Path, default=RESULTS / "spread_latency.json")
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

    rules = rules_for(n_vlm)
    results = {"meta": run_metadata(args), "by_k": {}}

    print(f"\n{'K':<5}{'rule':<13}{'max_gap':<10}{'clip':<10}{'latency s':<12}{'vs bisection'}")
    for k in args.k_values:
        results["by_k"][str(k)] = {}
        base = {}
        for name, fn in rules.items():
            idx = fn(k)
            gap = max_gap(idx, n_vlm)
            per_clip = {}
            for clip in CLIPS:
                inputs = prepare_inputs_for_clip(loaded, clip, device)
                pipe = switch(idx)
                for _ in range(args.warmup):
                    pipe.start_iteration()
                    with torch.no_grad():
                        adapter.run(loaded, inputs, a)
                    torch.cuda.synchronize()
                ts = []
                for _ in range(args.reps):
                    t0 = time.perf_counter()
                    pipe.start_iteration()
                    with torch.no_grad():
                        adapter.run(loaded, inputs, a)
                    torch.cuda.synchronize()
                    ts.append(time.perf_counter() - t0)
                pipe.remove()
                del pipe
                gc.collect()
                mean = statistics.mean(ts)
                per_clip[clip] = {"mean": mean, "times": ts}
                if name == "bisection":
                    base[clip] = mean
                delta = ""
                if clip in base and name != "bisection":
                    delta = f"{100 * (mean - base[clip]) / base[clip]:+.2f}%"
                print(f"{k:<5}{name:<13}{gap:<10}{clip[:8]:<10}{mean:<12.3f}{delta}")
            results["by_k"][str(k)][name] = {"max_gap": gap, "clips": per_clip}

    args.output.write_text(json.dumps(results, indent=2))

    print("\n=== Latency against spread, averaged over clips ===")
    print(f"{'K':<5}{'rule':<13}{'max_gap':<10}{'mean latency':<15}{'vs bisection'}")
    for k in args.k_values:
        blk = results["by_k"][str(k)]
        bis = statistics.mean(v["mean"] for v in blk["bisection"]["clips"].values())
        for name in ("sequential", "bisection", "greedy", "upstream"):
            m = statistics.mean(v["mean"] for v in blk[name]["clips"].values())
            print(f"{k:<5}{name:<13}{blk[name]['max_gap']:<10}{m:<15.3f}"
                  f"{100 * (m - bis) / bis:+.2f}%")

    print("\nIf sequential, whose spread is several times worse, is not markedly")
    print("slower, then spread is not what drives latency and the even spacing")
    print("the upstream rule maintains is not buying the thing it was built for.")
    print(f"\nSaved -> {args.output}")


if __name__ == "__main__":
    main()
