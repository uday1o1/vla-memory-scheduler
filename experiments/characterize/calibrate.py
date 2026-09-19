"""Calibrate latency(K) for the deadline-aware residency policy.

The oom-free-alpamayo profiler only computes max resident count from VRAM
capacity (budget / layer_size); it never measures per-layer timing. Our
adaptive policy needs an empirical latency(K) relationship to invert for
"what K meets this deadline", so we measure it directly: run N calls at
each of several K values, same clip, no contention, and record mean latency.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch  # noqa: E402

from pathlib import Path as _Path  # noqa: E402
sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))

from ias.paths import R1_CONFIG, RESULTS, bootstrap  # noqa: E402

bootstrap()

from ias.inputs import prepare_inputs_for_clip  # noqa: E402

from alpamayo_memopt import load_config  # noqa: E402
from alpamayo_memopt.models import TriHookPipeline, get_adapter  # noqa: E402





def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=R1_CONFIG)
    p.add_argument("--clip-id", default="d497f01b-4f68-4c27-9a6c-55872a1d6bd6")
    p.add_argument("--k-values", type=int, nargs="+", default=[0, 8, 16, 24, 30, 33])
    p.add_argument("--reps", type=int, default=5)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--output", type=Path, default=RESULTS / "calibration.json")
    p.add_argument("--device", type=int, default=0)
    args = p.parse_args()

    device = f"cuda:{args.device}"
    torch.cuda.set_device(args.device)

    from transformers.utils import logging as _hf
    _hf.set_verbosity_error()
    _hf.disable_progress_bar()

    config = load_config(args.config)
    adapter = get_adapter(config.model.kind)

    class Args:
        pass
    a = Args()
    a.model_id = None
    a.model_cache_dir = None
    a.model_revision = None
    a.attn_implementation = None
    a.local_files_only = False
    a.num_traj_samples = adapter.default_num_traj_samples
    a.max_generation_length = adapter.default_max_generation_length

    print("Loading model (once, reused across all K values)...")
    loaded = adapter.load(a, config)
    adapter.setup_essentials(loaded, device)
    inputs = prepare_inputs_for_clip(loaded, args.clip_id, device)
    n_vlm = len(loaded.vlm_layers)

    # Track which layers are currently on GPU so we only move deltas.
    current_resident = set()

    results = {}
    for k in args.k_values:
        from alpamayo_memopt.profiler import interleaved_placement
        target_indices = set(interleaved_placement(k, n_vlm)) if k > 0 else set()

        # Move newly-resident layers to GPU, newly-offloaded back to CPU.
        for i in target_indices - current_resident:
            loaded.vlm_layers[i].to(device)
        for i in current_resident - target_indices:
            loaded.vlm_layers[i].to("cpu")
        current_resident = target_indices
        torch.cuda.synchronize(args.device)

        pipeline = TriHookPipeline(loaded.vlm_layers, loaded.vit_blocks,
                                    loaded.expert_layers, list(target_indices), device=device)

        def run_once():
            pipeline.start_iteration()
            with torch.no_grad():
                adapter.run(loaded, inputs, a)
            torch.cuda.synchronize(args.device)

        for _ in range(args.warmup):
            run_once()

        times = []
        for _ in range(args.reps):
            t0 = time.perf_counter()
            run_once()
            times.append(time.perf_counter() - t0)

        pipeline.remove()
        mean_s = sum(times) / len(times)
        print(f"K={k:2d} resident: mean={mean_s:.3f}s  times={[f'{t:.3f}' for t in times]}")
        results[str(k)] = {"mean_s": mean_s, "times_s": times, "resident_indices": sorted(target_indices)}
        gc.collect()
        torch.cuda.empty_cache()

    args.output.write_text(json.dumps({
        "clip_id": args.clip_id,
        "n_vlm_layers": n_vlm,
        "calibration": results,
    }, indent=2))
    print(f"\nSaved calibration -> {args.output}")


if __name__ == "__main__":
    main()
