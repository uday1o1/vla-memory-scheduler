"""Validate that inference actually works AFTER a residency transition.

This path was never exercised: in the real run the adaptive arm never
changed K (hysteresis held it), so a post-transition inference call has
never been proven to work. A probe hit a device-mismatch error here.

Also checks whether interleaved_placement sets are nested across K, since
a non-nested placement means a transition must move layers in BOTH
directions or some "resident" layer is left stranded on the CPU.
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch

from pathlib import Path as _Path  # noqa: E402
sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))

from ias.paths import R1_CONFIG, bootstrap  # noqa: E402

bootstrap()
from alpamayo_memopt import load_config
from alpamayo_memopt.models import TriHookPipeline, get_adapter
from alpamayo_memopt.profiler import interleaved_placement
from ias.inputs import prepare_inputs_for_clip


def check_nesting():
    print("=== Are interleaved_placement sets nested across K? ===")
    prev_k, prev_set = None, None
    non_nested = []
    for k in (8, 16, 20, 24, 30, 33):
        s = set(interleaved_placement(k, 36))
        if prev_set is not None:
            missing = prev_set - s
            added = s - prev_set
            if missing:
                non_nested.append((prev_k, k, sorted(missing)))
            print(f"  K={prev_k}->{k}: +{len(added)} added, -{len(missing)} REMOVED "
                  f"{'(NOT nested)' if missing else '(nested)'}")
        prev_k, prev_set = k, s
    if non_nested:
        print("  => Placements are NOT nested. Transitions MUST move both directions.")
    else:
        print("  => Placements are nested (growing K only adds layers).")
    return len(non_nested) == 0


def main():
    nested = check_nesting()

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

    print("\n=== Post-transition inference validation ===")
    print("Loading model...")
    loaded = adapter.load(a, config)
    adapter.setup_essentials(loaded, device)
    n_vlm = len(loaded.vlm_layers)
    inputs = prepare_inputs_for_clip(loaded, "d497f01b-4f68-4c27-9a6c-55872a1d6bd6", device)

    current = set(interleaved_placement(33, n_vlm))
    for i in current:
        loaded.vlm_layers[i].to(device)
    torch.cuda.synchronize()
    pipe = TriHookPipeline(loaded.vlm_layers, loaded.vit_blocks,
                            loaded.expert_layers, list(current), device=device)

    def run_call(p):
        p.start_iteration()
        with torch.no_grad():
            adapter.run(loaded, inputs, a)
        torch.cuda.synchronize()

    import time
    t0 = time.perf_counter()
    run_call(pipe)
    print(f"  K=33 baseline call: {time.perf_counter()-t0:.3f}s  OK")

    # Exercise the real transition path in BOTH directions, with a real call after each.
    all_ok = True
    for target_k in (24, 16, 24, 33):
        target = set(interleaved_placement(target_k, n_vlm))
        try:
            pipe.remove()
            for i in target - current:
                loaded.vlm_layers[i].to(device)
            for i in current - target:
                loaded.vlm_layers[i].to("cpu")
            torch.cuda.synchronize()
            pipe = TriHookPipeline(loaded.vlm_layers, loaded.vit_blocks,
                                    loaded.expert_layers, list(target), device=device)
            current = target

            # Verify no resident layer was stranded on the CPU.
            stranded = [i for i in sorted(current)
                        if next(loaded.vlm_layers[i].parameters()).device.type != "cuda"]
            if stranded:
                print(f"  K->{target_k}: STRANDED resident layers on CPU: {stranded}")
                all_ok = False
                continue

            t0 = time.perf_counter()
            run_call(pipe)
            print(f"  K->{target_k}: transition + call OK ({time.perf_counter()-t0:.3f}s)")
        except Exception as e:
            print(f"  K->{target_k}: FAILED {type(e).__name__}: {str(e)[:160]}")
            all_ok = False

    pipe.remove()
    print(f"\n  VERDICT: all transitions valid = {all_ok}")


if __name__ == "__main__":
    main()
