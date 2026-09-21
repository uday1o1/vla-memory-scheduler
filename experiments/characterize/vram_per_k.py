"""Clean measurement of actual VRAM footprint per residency level K.

Decides whether co-location (multiple VLA instances sharing one GPU at
reduced K) is feasible - the number that determines which redesign is
strongest. Earlier probe numbers were contaminated by allocator caching
across successive K attempts; this measures each K from a clean state and
runs a real inference call to capture peak, not just resident, footprint.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import gc
import os
import sys

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch

from pathlib import Path as _Path  # noqa: E402
sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))

from scheduler.paths import R1_CONFIG, RESULTS, bootstrap  # noqa: E402

bootstrap()
from alpamayo_memopt import load_config
from alpamayo_memopt.models import TriHookPipeline, get_adapter
from alpamayo_memopt.profiler import interleaved_placement
from scheduler.inputs import prepare_inputs_for_clip
from scheduler.provenance import run_metadata


def gb(x):
    return x / (1024 ** 3)


def main():
    # K values are card specific. A 16GB card cannot reach the counts a 24GB
    # card can, so hardcoding them would crash on smaller hardware.
    p = argparse.ArgumentParser()
    p.add_argument("--k-values", type=int, nargs="+", default=[33, 24, 16, 8])
    p.add_argument("--output", type=Path, default=RESULTS / "vram_per_k.json",
                   help="footprint per residency level, machine readable")
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

    total = torch.cuda.get_device_properties(0).total_memory
    print(f"GPU total: {gb(total):.2f}GB")

    loaded = adapter.load(a, config)
    adapter.setup_essentials(loaded, device)
    n_vlm = len(loaded.vlm_layers)
    inputs = prepare_inputs_for_clip(loaded, "d497f01b-4f68-4c27-9a6c-55872a1d6bd6", device)

    torch.cuda.synchronize()
    free_ess, _ = torch.cuda.mem_get_info()
    print(f"Essentials + inputs only (K=0 resident): uses {gb(total - free_ess):.2f}GB\n")

    measured: dict[str, dict] = {}
    print(f"{'K':<6}{'peak_alloc_GB':<16}{'process_used_GB':<18}{'2x fits?'}")
    current = set()
    for k in args.k_values:
        target = set(interleaved_placement(k, n_vlm)) if k > 0 else set()
        for i in target - current:
            loaded.vlm_layers[i].to(device)
        for i in current - target:
            loaded.vlm_layers[i].to("cpu")
        current = target
        gc.collect(); torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

        pipe = TriHookPipeline(loaded.vlm_layers, loaded.vit_blocks,
                                loaded.expert_layers, list(target), device=device)
        pipe.start_iteration()
        with torch.no_grad():
            adapter.run(loaded, inputs, a)
        torch.cuda.synchronize()

        peak = torch.cuda.max_memory_allocated()
        free_now, _ = torch.cuda.mem_get_info()
        process_used = total - free_now
        pipe.remove()
        fits2x = (2 * process_used) < total
        measured[str(k)] = {
            "peak_alloc_gb": gb(peak),
            "process_used_gb": gb(process_used),
            "two_fit": bool(fits2x),
        }
        print(f"{k:<6}{gb(peak):<16.2f}{gb(process_used):<18.2f}{fits2x}")

    print("\nNote: process_used includes the allocator's cached reservation,")
    print("which is what a co-located process actually cannot take.")

    args.output.write_text(json.dumps({
        "meta": run_metadata(args),
        "gpu_total_gb": gb(total),
        "by_k": measured,
    }, indent=2))
    print(f"\nSaved -> {args.output}")


if __name__ == "__main__":
    main()
