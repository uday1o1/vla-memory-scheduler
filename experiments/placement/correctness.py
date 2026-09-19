"""Correctness guardrail: does placement change model output?

Residency placement decides where weights live, not what is computed, so
upstream and nested placement at the same K should produce identical
trajectories. That is the expectation, not a measurement, until it is
checked. A performance optimization that silently changes outputs is not a
performance optimization.

Compares predicted trajectories element-wise at several residency levels and
across clips, including the K=0 fully-streamed case as a control.
"""
from __future__ import annotations

import json
import os
import sys
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

CLIPS = [
    "d497f01b-4f68-4c27-9a6c-55872a1d6bd6",
    "441057af-5c65-4d8e-993d-713090072248",
    "59aba96d-8920-4e49-8edb-c08bd800edf1",
]
KS = [16, 24, 33]


def extract_traj(out):
    """Pull a comparable float tensor out of the adapter's return value."""
    def walk(x):
        if isinstance(x, torch.Tensor) and x.is_floating_point() and x.numel() > 1:
            return x.detach().float().cpu()
        if isinstance(x, dict):
            for v in x.values():
                r = walk(v)
                if r is not None:
                    return r
        if isinstance(x, (list, tuple)):
            for v in x:
                r = walk(v)
                if r is not None:
                    return r
        return None
    return walk(out)


def main():
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

    def run(pipe):
        pipe.start_iteration()
        with torch.no_grad():
            out = adapter.run(loaded, inputs, a)
        torch.cuda.synchronize()
        return extract_traj(out)

    results = {}
    print(f"\n{'K':<6}{'clip':<10}{'identical':<12}{'max abs diff':<16}{'shape'}")
    all_identical = True
    for k in KS:
        for clip in CLIPS:
            inputs = prepare_inputs_for_clip(loaded, clip, device)

            p = switch(interleaved_placement(k, n_vlm))
            out_u = run(p)
            p.remove()

            p = switch(nested_placement(k, n_vlm))
            out_n = run(p)
            p.remove()

            if out_u is None or out_n is None:
                print(f"{k:<6}{clip[:8]:<10}{'n/a':<12}{'could not extract tensor':<16}")
                continue

            identical = torch.equal(out_u, out_n)
            max_diff = (out_u - out_n).abs().max().item() if out_u.shape == out_n.shape else float("nan")
            all_identical &= identical
            results.setdefault(str(k), {})[clip] = {
                "identical": bool(identical), "max_abs_diff": max_diff,
                "shape": list(out_u.shape),
            }
            print(f"{k:<6}{clip[:8]:<10}{str(identical):<12}{max_diff:<16.3e}{list(out_u.shape)}")

    RESULTS / "placement_correctness.json".write_text(json.dumps(results, indent=2))
    print(f"\nAll outputs bit-identical across placements: {all_identical}")
    if not all_identical:
        print("Any nonzero difference would need explaining before the latency")
        print("claim means anything, since the two rules would not be computing")
        print("the same thing.")
    print("\nSaved -> /root/placement_correctness.json")


if __name__ == "__main__":
    main()
