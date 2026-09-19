"""Does nested placement cost anything in steady-state latency?

The nested rule matches upstream's spread metric exactly at every K while
needing far fewer layer movements per residency change. If it also matches
on real inference latency, it strictly dominates for adaptive residency.
This measures both: steady-state latency per K under each rule, and the
wall-clock transition cost of an adaptation path under each rule.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch

sys.path.insert(0, "/root/oom-free-alpamayo")
sys.path.insert(0, "/root")
from alpamayo_memopt import load_config
from alpamayo_memopt.models import TriHookPipeline, get_adapter
from alpamayo_memopt.profiler import interleaved_placement
from ias_calibrate import prepare_inputs_for_clip
from ias_placement import nested_placement

K_VALUES = [16, 24, 30, 33]
PATH = [33, 24, 30, 16, 33]  # representative adaptation path


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

    print("Loading model...")
    loaded = adapter.load(a, config)
    adapter.setup_essentials(loaded, device)
    n_vlm = len(loaded.vlm_layers)
    inputs = prepare_inputs_for_clip(loaded, "d497f01b-4f68-4c27-9a6c-55872a1d6bd6", device)

    current: set[int] = set()

    def switch_to(indices: list[int]):
        nonlocal current
        target = set(indices)
        for i in target - current:
            loaded.vlm_layers[i].to(device)
        for i in current - target:
            loaded.vlm_layers[i].to("cpu")
        torch.cuda.synchronize()
        current = target
        return TriHookPipeline(loaded.vlm_layers, loaded.vit_blocks,
                                loaded.expert_layers, list(target), device=device)

    def timed_call(pipe, reps=3):
        pipe.start_iteration()
        with torch.no_grad():
            adapter.run(loaded, inputs, a)
        torch.cuda.synchronize()
        ts = []
        for _ in range(reps):
            t0 = time.perf_counter()
            pipe.start_iteration()
            with torch.no_grad():
                adapter.run(loaded, inputs, a)
            torch.cuda.synchronize()
            ts.append(time.perf_counter() - t0)
        return sum(ts) / len(ts)

    results = {"latency": {}, "transition": {}}

    print(f"\n=== Steady-state latency per K ===")
    print(f"{'K':<6}{'upstream (s)':<16}{'nested (s)':<16}{'delta'}")
    for k in K_VALUES:
        p = switch_to(interleaved_placement(k, n_vlm))
        lat_u = timed_call(p)
        p.remove()
        p = switch_to(nested_placement(k, n_vlm))
        lat_n = timed_call(p)
        p.remove()
        results["latency"][k] = {"upstream": lat_u, "nested": lat_n}
        print(f"{k:<6}{lat_u:<16.3f}{lat_n:<16.3f}{lat_n - lat_u:+.3f}")

    print(f"\n=== Wall-clock transition cost over path {PATH} ===")
    for name, rule in (("upstream", interleaved_placement), ("nested", nested_placement)):
        p = switch_to(rule(PATH[0], n_vlm))
        total_s, total_moves = 0.0, 0
        for k in PATH[1:]:
            target = set(rule(k, n_vlm))
            moves = len(target - current) + len(current - target)
            p.remove()
            t0 = time.perf_counter()
            p = switch_to(sorted(target))
            total_s += time.perf_counter() - t0
            total_moves += moves
        p.remove()
        results["transition"][name] = {"seconds": total_s, "moves": total_moves}
        print(f"  {name:<10} {total_moves:>3} moves   {total_s:>7.3f}s")

    u = results["transition"]["upstream"]
    n = results["transition"]["nested"]
    print(f"\n  nested saves {u['seconds'] - n['seconds']:.3f}s "
          f"({u['seconds'] / n['seconds']:.2f}x less transition time) "
          f"with {u['moves'] / n['moves']:.2f}x fewer moves")

    Path("/root/placement_eval.json").write_text(json.dumps(results, indent=2))
    print("\nSaved -> /root/placement_eval.json")


if __name__ == "__main__":
    main()
