"""Bounded experiment: does compute contention affect latency(K) uniformly,
or does it disproportionately hurt low-K (streaming-heavy) vs high-K
(residency-heavy) configurations?

This resolves a real design fork in Algorithm 1.2: should the policy
increase or decrease K under sensed contention? Answered empirically here
rather than assumed.
"""
from __future__ import annotations

import gc
import multiprocessing as mp
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


def dummy_compute_load(stop_flag, device_idx: int = 0, duty_cycle: float = 0.70):
    """Background compute-bound kernel at our locked 70% duty-cycle target."""
    sys.path.insert(0, "/root")
    from ias_contention_generator import steady_load
    while not stop_flag.is_set():
        steady_load(time.time() + 1.0, duty_cycle=duty_cycle, device_idx=device_idx)


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
    inputs = prepare_inputs_for_clip(loaded, "d497f01b-4f68-4c27-9a6c-55872a1d6bd6", device)
    n_vlm = len(loaded.vlm_layers)

    test_k_values = [16, 24, 33]
    current_resident = set()

    print(f"\n{'K':<6}{'No-contention (s)':<20}{'With-contention (s)':<22}{'Slowdown factor'}")
    for k in test_k_values:
        target_indices = set(interleaved_placement(k, n_vlm)) if k > 0 else set()
        for i in target_indices - current_resident:
            loaded.vlm_layers[i].to(device)
        for i in current_resident - target_indices:
            loaded.vlm_layers[i].to("cpu")
        current_resident = target_indices
        torch.cuda.synchronize()

        pipeline = TriHookPipeline(loaded.vlm_layers, loaded.vit_blocks,
                                    loaded.expert_layers, list(target_indices), device=device)

        def run_once():
            pipeline.start_iteration()
            with torch.no_grad():
                adapter.run(loaded, inputs, a)
            torch.cuda.synchronize()

        run_once()  # warmup

        # No contention
        t0 = time.perf_counter()
        run_once()
        no_contention_s = time.perf_counter() - t0

        # With contention: launch a real second process under MPS
        stop_flag = mp.Event()
        proc = mp.Process(target=dummy_compute_load, args=(stop_flag, 0))
        proc.start()
        time.sleep(1.0)  # let contention ramp up

        t0 = time.perf_counter()
        run_once()
        with_contention_s = time.perf_counter() - t0

        stop_flag.set()
        proc.join(timeout=5)
        if proc.is_alive():
            proc.terminate()

        slowdown = with_contention_s / no_contention_s
        print(f"{k:<6}{no_contention_s:<20.3f}{with_contention_s:<22.3f}{slowdown:.3f}x")

        pipeline.remove()
        gc.collect()
        torch.cuda.empty_cache()
        time.sleep(1.0)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
