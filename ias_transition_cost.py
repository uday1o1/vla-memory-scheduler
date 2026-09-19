"""Measure the cost of CHANGING residency mid-run (not the decision cost).

This isolates: time to move layers CPU<->GPU + re-pin host memory +
reconstruct TriHookPipeline, separate from the cheap binary-search decision.
"""
from __future__ import annotations

import gc
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch  # noqa: E402

sys.path.insert(0, "/root/oom-free-alpamayo")

from alpamayo_memopt import load_config  # noqa: E402
from alpamayo_memopt.models import TriHookPipeline, get_adapter  # noqa: E402
from alpamayo_memopt.profiler import interleaved_placement  # noqa: E402

sys.path.insert(0, "/root")
from ias_calibrate import prepare_inputs_for_clip  # noqa: E402


def rebuild_pipeline(loaded, old_resident: set, new_resident: set, device: str):
    """Move deltas, rebuild pipeline. Returns (move_s, rebuild_s)."""
    t0 = time.perf_counter()
    for i in new_resident - old_resident:
        loaded.vlm_layers[i].to(device)
    for i in old_resident - new_resident:
        loaded.vlm_layers[i].to("cpu")
    torch.cuda.synchronize()
    move_s = time.perf_counter() - t0

    t1 = time.perf_counter()
    pipeline = TriHookPipeline(loaded.vlm_layers, loaded.vit_blocks,
                                loaded.expert_layers, list(new_resident), device=device)
    torch.cuda.synchronize()
    rebuild_s = time.perf_counter() - t1
    return pipeline, move_s, rebuild_s


def main():
    device = "cuda:0"
    torch.cuda.set_device(0)
    from transformers.utils import logging as _hf
    _hf.set_verbosity_error()
    _hf.disable_progress_bar()

    config = load_config(Path("/root/oom-free-alpamayo/r1_config.json"))
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

    print("Loading model...")
    loaded = adapter.load(a, config)
    adapter.setup_essentials(loaded, device)
    n_vlm = len(loaded.vlm_layers)

    # Start at K=33 (from profiler), then test transitions to various targets.
    start_k = 33
    resident = set(interleaved_placement(start_k, n_vlm))
    for i in resident:
        loaded.vlm_layers[i].to(device)
    torch.cuda.synchronize()
    pipeline = TriHookPipeline(loaded.vlm_layers, loaded.vit_blocks,
                                loaded.expert_layers, list(resident), device=device)

    test_transitions = [(33, 28), (28, 33), (33, 20), (20, 33), (33, 32), (32, 33)]
    print(f"\nStarting resident: K={start_k}")
    print(f"{'From->To':<12} {'#Delta layers':<15} {'Move(s)':<10} {'Rebuild(s)':<12} {'Total(s)'}")
    current_resident = resident
    for from_k, to_k in test_transitions:
        new_resident = set(interleaved_placement(to_k, n_vlm))
        n_delta = len(current_resident.symmetric_difference(new_resident))
        pipeline.remove()
        pipeline, move_s, rebuild_s = rebuild_pipeline(loaded, current_resident, new_resident, device)
        current_resident = new_resident
        print(f"{from_k}->{to_k:<9} {n_delta:<15} {move_s:<10.4f} {rebuild_s:<12.4f} {move_s + rebuild_s:.4f}")

    pipeline.remove()


if __name__ == "__main__":
    main()
