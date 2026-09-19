"""Decisive probes for the memory-pressure redesign.

Three questions that determine whether the redesign is viable at all:
  A. Does torch.cuda.mem_get_info() see ANOTHER process's allocation?
  B. With the model already loaded at K=33, what happens when a second
     process grabs VRAM - do we fail, slow down, or continue unaffected?
  C. If a second process holds VRAM FIRST, can K=33 still load? Does a
     lower K load successfully in the same conditions?

C is the crux: it establishes whether a regime exists where the static
policy fails and an adaptive one survives.
"""
from __future__ import annotations

import multiprocessing as mp
import os
import sys
import time

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch

from pathlib import Path as _Path  # noqa: E402
sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))

from ias.paths import R1_CONFIG, bootstrap  # noqa: E402

bootstrap()

from ias.memory import gb, hog_vram  # noqa: E402


def probe_a():
    print("\n=== A. Does mem_get_info see another process's allocation? ===")
    free0, total0 = torch.cuda.mem_get_info()
    print(f"  Before: free={gb(free0):.2f}GB / total={gb(total0):.2f}GB")

    ready = mp.Event()
    p = mp.Process(target=hog_vram, args=(6.0, 12.0, ready))
    p.start()
    ready.wait(timeout=60)
    time.sleep(2.0)

    free1, _ = torch.cuda.mem_get_info()
    print(f"  With 6GB hog running: free={gb(free1):.2f}GB")
    delta = gb(free0) - gb(free1)
    print(f"  Observed drop: {delta:.2f}GB -> mem_get_info IS cross-process aware: {delta > 4.0}")

    p.terminate(); p.join(timeout=10)
    time.sleep(2.0)
    free2, _ = torch.cuda.mem_get_info()
    print(f"  After hog terminated: free={gb(free2):.2f}GB (recovered: {gb(free2) - gb(free1):.2f}GB)")
    return delta > 4.0


def probe_c():
    """Can the model load at K=33 when VRAM is already held by another process?"""
    print("\n=== C. Can K=33 load under pre-existing memory pressure? ===")
    from alpamayo_memopt import load_config
    from alpamayo_memopt.models import TriHookPipeline, get_adapter
    from alpamayo_memopt.profiler import interleaved_placement

    from transformers.utils import logging as _hf
    _hf.set_verbosity_error(); _hf.disable_progress_bar()

    config = load_config(R1_CONFIG)
    adapter = get_adapter(config.model.kind)

    class A: pass
    a = A()
    a.model_id = None; a.model_cache_dir = None; a.model_revision = None
    a.attn_implementation = None; a.local_files_only = False
    a.num_traj_samples = adapter.default_num_traj_samples
    a.max_generation_length = adapter.default_max_generation_length

    hold = 10.0
    print(f"  Starting a {hold}GB VRAM hog FIRST...")
    ready = mp.Event()
    p = mp.Process(target=hog_vram, args=(hold, 600.0, ready))
    p.start()
    ready.wait(timeout=120)
    time.sleep(3.0)

    free, total = torch.cuda.mem_get_info()
    print(f"  Free VRAM now: {gb(free):.2f}GB / {gb(total):.2f}GB")

    device = "cuda:0"
    print("  Loading model to CPU...")
    loaded = adapter.load(a, config)
    adapter.setup_essentials(loaded, device)
    torch.cuda.synchronize()
    free_after_ess, _ = torch.cuda.mem_get_info()
    print(f"  After essentials on GPU: free={gb(free_after_ess):.2f}GB")

    n_vlm = len(loaded.vlm_layers)
    for k in (33, 24, 16, 8):
        print(f"  Trying K={k}...")
        try:
            target = set(interleaved_placement(k, n_vlm))
            for i in target:
                loaded.vlm_layers[i].to(device)
            torch.cuda.synchronize()
            pipe = TriHookPipeline(loaded.vlm_layers, loaded.vit_blocks,
                                    loaded.expert_layers, list(target), device=device)
            inputs_ok = True
            free_k, _ = torch.cuda.mem_get_info()
            print(f"    K={k} LOADED ok (free after: {gb(free_k):.2f}GB)")
            pipe.remove()
            for i in target:
                loaded.vlm_layers[i].to("cpu")
            torch.cuda.synchronize()
        except Exception as e:
            print(f"    K={k} FAILED: {type(e).__name__}: {str(e)[:160]}")
            for i in list(target):
                try:
                    loaded.vlm_layers[i].to("cpu")
                except Exception:
                    pass
            torch.cuda.empty_cache()

    p.terminate(); p.join(timeout=10)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    torch.cuda.set_device(0)
    cross_process_aware = probe_a()
    if cross_process_aware:
        probe_c()
    else:
        print("\nmem_get_info is NOT cross-process aware - redesign sensing mechanism is invalid.")
