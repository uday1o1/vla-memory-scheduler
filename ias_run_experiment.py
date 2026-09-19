"""Full experiment runner: 6 reps x 3 scenarios x 2 methods x N calls.

Contention sensing: since DCGM's Profiling module (fine-grained SM-occupancy
counters) fails to load in this container, and nvidia-smi's utilization.gpu
is too coarse (see PROJECT_DETAILS.md), the adaptive policy senses contention
from its OWN recent latency drift - a rolling window of observed call
latencies compared against the no-contention calibration for the current K.
This is a realistic, privilege-free signal: no real deployed system would
assume access to hardware profiling counters either.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
from collections import deque
from pathlib import Path

import torch

sys.path.insert(0, "/root/oom-free-alpamayo")
from alpamayo_memopt import load_config  # noqa: E402
from alpamayo_memopt.models import TriHookPipeline, get_adapter  # noqa: E402
from alpamayo_memopt.profiler import interleaved_placement  # noqa: E402

sys.path.insert(0, "/root")
from ias_calibrate import prepare_inputs_for_clip  # noqa: E402
from ias_policy import CALIBRATION, PolicyState, decide_residency  # noqa: E402
from ias_contention_generator import steady_load, bursty_load  # noqa: E402

CLIPS = [
    "d497f01b-4f68-4c27-9a6c-55872a1d6bd6",
    "441057af-5c65-4d8e-993d-713090072248",
    "59aba96d-8920-4e49-8edb-c08bd800edf1",
    "8825b1fa-0abf-4a28-b444-25fef71cadcb",
    "89e4e404-80ea-42e2-bf8e-8395af5ab111",
    "7d109673-d967-4b02-93c5-6d2d25d964d0",
]
MAX_K = 33
N_VLM = 36
ROLLING_WINDOW = 5
DRIFT_THRESHOLD = 1.01  # recent avg > 1% slower than calibration -> sensed contention


def rebuild_pipeline(loaded, old_resident: set, new_resident: set, device: str):
    for i in new_resident - old_resident:
        loaded.vlm_layers[i].to(device)
    for i in old_resident - new_resident:
        loaded.vlm_layers[i].to("cpu")
    torch.cuda.synchronize()
    return TriHookPipeline(loaded.vlm_layers, loaded.vit_blocks,
                            loaded.expert_layers, list(new_resident), device=device)


def sensed_gpu_util(recent_latencies: deque, current_k: int) -> float:
    """Self-referential contention proxy: are recent calls slower than
    the no-contention calibration predicts for the current K?"""
    if len(recent_latencies) < ROLLING_WINDOW:
        return 0.0
    avg = sum(recent_latencies) / len(recent_latencies)
    expected = CALIBRATION.latency_at(current_k)
    return 1.0 if avg > expected * DRIFT_THRESHOLD else 0.0


def run_calls(loaded, adapter, a, inputs, device, method: str, deadline: float,
              n_calls: int) -> dict:
    current_resident = set(interleaved_placement(MAX_K, N_VLM))
    for i in current_resident:
        loaded.vlm_layers[i].to(device)
    torch.cuda.synchronize()
    pipeline = TriHookPipeline(loaded.vlm_layers, loaded.vit_blocks,
                                loaded.expert_layers, list(current_resident), device=device)

    state = PolicyState(current_k=MAX_K, max_k=MAX_K)
    recent = deque(maxlen=ROLLING_WINDOW)
    latencies, residency_ks, transition_times = [], [], []

    for _ in range(n_calls):
        transition_s = 0.0
        if method == "adaptive":
            gpu_util = sensed_gpu_util(recent, state.current_k)
            new_k = decide_residency(state, deadline_s=deadline, gpu_util=gpu_util)
            if new_k != state.current_k:
                new_resident = set(interleaved_placement(new_k, N_VLM))
                t0 = time.perf_counter()
                pipeline.remove()
                pipeline = rebuild_pipeline(loaded, current_resident, new_resident, device)
                transition_s = time.perf_counter() - t0
                current_resident = new_resident
                state.current_k = new_k

        t0 = time.perf_counter()
        pipeline.start_iteration()
        with torch.no_grad():
            adapter.run(loaded, inputs, a)
        torch.cuda.synchronize()
        latency = time.perf_counter() - t0

        latencies.append(latency)
        residency_ks.append(state.current_k)
        transition_times.append(transition_s)
        recent.append(latency)

    pipeline.remove()
    return {"latencies": latencies, "residency_k": residency_ks,
            "transition_s": transition_times}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-calls", type=int, default=100)
    p.add_argument("--reps", type=int, default=6)
    p.add_argument("--output", type=Path, default=Path("/root/results.json"))
    p.add_argument("--device", type=int, default=0)
    args = p.parse_args()

    device = f"cuda:{args.device}"
    torch.cuda.set_device(args.device)
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

    print("Loading model (once, reused across all reps)...")
    loaded = adapter.load(a, config)
    adapter.setup_essentials(loaded, device)

    all_results = {}
    for rep_idx in range(args.reps):
        clip_id = CLIPS[rep_idx]
        print(f"\n=== Rep {rep_idx + 1}/{args.reps} (clip {clip_id}) ===")
        inputs = prepare_inputs_for_clip(loaded, clip_id, device)

        # Scenario 1: no contention. Baseline run also gives us this rep's deadline.
        print("  [no-contention, baseline]")
        r = run_calls(loaded, adapter, a, inputs, device, "baseline", deadline=None,
                       n_calls=args.n_calls)
        latencies_sorted = sorted(r["latencies"])
        p75 = latencies_sorted[int(0.75 * len(latencies_sorted))]
        deadline = p75 * 1.15
        print(f"    p75={p75:.3f}s, deadline={deadline:.3f}s")
        all_results.setdefault(clip_id, {})["no_contention_baseline"] = r
        all_results[clip_id]["deadline"] = deadline

        print("  [no-contention, adaptive]")
        r = run_calls(loaded, adapter, a, inputs, device, "adaptive", deadline=deadline,
                       n_calls=args.n_calls)
        all_results[clip_id]["no_contention_adaptive"] = r

        # Scenario 2: steady contention (separate process, 70% duty cycle, locked target)
        for method in ("baseline", "adaptive"):
            print(f"  [steady-contention, {method}]")
            stop_flag = mp.Event()
            proc = mp.Process(target=_steady_contention_proc, args=(args.device,))
            proc.start()
            time.sleep(1.0)
            r = run_calls(loaded, adapter, a, inputs, device, method,
                           deadline=deadline, n_calls=args.n_calls)
            proc.terminate()
            proc.join(timeout=5)
            all_results[clip_id][f"steady_{method}"] = r

        # Scenario 3: bursty contention (Poisson, 500ms bursts, locked params)
        for method in ("baseline", "adaptive"):
            print(f"  [bursty-contention, {method}]")
            proc = mp.Process(target=_bursty_contention_proc,
                               args=(args.device, args.n_calls * 8))  # long enough to cover the run
            proc.start()
            time.sleep(0.5)
            r = run_calls(loaded, adapter, a, inputs, device, method,
                           deadline=deadline, n_calls=args.n_calls)
            proc.terminate()
            proc.join(timeout=5)
            all_results[clip_id][f"bursty_{method}"] = r

        # Save incrementally after each rep so a crash doesn't lose prior progress.
        args.output.write_text(json.dumps(all_results, indent=2))
        print(f"  Saved progress -> {args.output}")

    print(f"\nAll {args.reps} reps complete -> {args.output}")


def _steady_contention_proc(device_idx: int):
    steady_load(time.time() + 3600, duty_cycle=0.70, device_idx=device_idx)


def _bursty_contention_proc(device_idx: int, duration_s: float):
    bursty_load(duration_s, burst_duration_s=0.5, lambda_rate=0.2, device_idx=device_idx)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
