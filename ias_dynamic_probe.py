"""Does mid-run shedding actually rescue a call when pressure arrives late?

Probe C established that a low K can LOAD under pre-existing pressure.
The real experimental path is different: already running at K=33, pressure
arrives, and the policy must shed in time to complete the next call.
PyTorch's caching allocator could behave differently in that order.
"""
from __future__ import annotations

import multiprocessing as mp
import os
import sys
import time

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch

sys.path.insert(0, "/root/oom-free-alpamayo")
sys.path.insert(0, "/root")
from alpamayo_memopt import load_config
from alpamayo_memopt.models import TriHookPipeline, get_adapter
from alpamayo_memopt.profiler import interleaved_placement
from ias_calibrate import prepare_inputs_for_clip
from ias_memprobe import hog_vram, gb


def main():
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

    print("Loading model at K=33 (clean GPU)...")
    loaded = adapter.load(a, config)
    adapter.setup_essentials(loaded, device)
    n_vlm = len(loaded.vlm_layers)
    resident = set(interleaved_placement(33, n_vlm))
    for i in resident:
        loaded.vlm_layers[i].to(device)
    torch.cuda.synchronize()
    pipe = TriHookPipeline(loaded.vlm_layers, loaded.vit_blocks,
                            loaded.expert_layers, list(resident), device=device)
    inputs = prepare_inputs_for_clip(loaded, "d497f01b-4f68-4c27-9a6c-55872a1d6bd6", device)

    def run_call():
        pipe.start_iteration()
        with torch.no_grad():
            adapter.run(loaded, inputs, a)
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    run_call()
    print(f"  Clean call at K=33: {time.perf_counter()-t0:.3f}s")
    free, _ = torch.cuda.mem_get_info()
    print(f"  Free VRAM: {gb(free):.2f}GB")

    print("\nPressure arriving (8GB hog) while loaded at K=33...")
    ready = mp.Event()
    p = mp.Process(target=hog_vram, args=(8.0, 600.0, ready))
    p.start()
    ready.wait(timeout=120)
    time.sleep(3.0)
    free, _ = torch.cuda.mem_get_info()
    print(f"  Free VRAM now: {gb(free):.2f}GB")

    print("  Attempting a call at K=33 under pressure (static-high behavior):")
    try:
        t0 = time.perf_counter()
        run_call()
        print(f"    SUCCEEDED in {time.perf_counter()-t0:.3f}s (no OOM - pressure insufficient?)")
        static_high_survived = True
    except Exception as e:
        print(f"    FAILED: {type(e).__name__}: {str(e)[:140]}")
        static_high_survived = False

    print("\n  Now shedding to K=20 (adaptive behavior) and retrying:")
    try:
        pipe.remove()
        target = set(interleaved_placement(20, n_vlm))
        t_shed = time.perf_counter()
        for i in resident - target:
            loaded.vlm_layers[i].to("cpu")
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        shed_s = time.perf_counter() - t_shed
        pipe2 = TriHookPipeline(loaded.vlm_layers, loaded.vit_blocks,
                                 loaded.expert_layers, list(target), device=device)
        free, _ = torch.cuda.mem_get_info()
        print(f"    Shed in {shed_s:.3f}s, free VRAM now {gb(free):.2f}GB")

        def run_call2():
            pipe2.start_iteration()
            with torch.no_grad():
                adapter.run(loaded, inputs, a)
            torch.cuda.synchronize()

        t0 = time.perf_counter()
        run_call2()
        print(f"    Call at K=20 under pressure SUCCEEDED in {time.perf_counter()-t0:.3f}s")
        adaptive_survived = True
    except Exception as e:
        print(f"    FAILED: {type(e).__name__}: {str(e)[:140]}")
        adaptive_survived = False

    p.terminate(); p.join(timeout=10)
    print(f"\n  VERDICT: static-high survived={static_high_survived}, "
          f"adaptive(shed) survived={adaptive_survived}")
    print(f"  Regime exists (static fails, adaptive survives): "
          f"{(not static_high_survived) and adaptive_survived}")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
