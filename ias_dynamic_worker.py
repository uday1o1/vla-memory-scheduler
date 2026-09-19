"""Adaptive profile switching when memory pressure changes during a run.

The direction matters. GPU memory is won by whoever allocates first, so a
competitor cannot squeeze a process that already holds its allocation. What
can happen is the reverse: the competitor releases, memory becomes available,
and a process running on a small profile could now afford a faster one.

So the transition this experiment exercises is the upgrade, and the question
the professor raised applies directly to it. Upgrading costs a residency
change. If that change costs more than it saves, the upgrade is a loss. This
measures whether it pays back, and how that depends on the placement rule
governing how many layers must move.

  fixed     stays on whatever profile fit at startup, never upgrades
  adaptive  re-checks free memory before each call and upgrades when a
            faster profile now fits
"""
from __future__ import annotations

import argparse
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
from ias_profiles import PROFILES, deadline_s, free_gb, select_profile


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--policy", choices=["fixed", "adaptive"], required=True)
    p.add_argument("--placement", choices=["nested", "upstream"], default="nested")
    p.add_argument("--n-calls", type=int, default=30)
    p.add_argument("--clip-id", required=True)
    p.add_argument("--release-after", type=int, default=10,
                   help="call index at which the external pressure is released")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--marker", type=Path, default=None,
                   help="touched once timed calls begin, so the driver can time "
                        "the pressure release without guessing model load time")
    p.add_argument("--safety-gb", type=float, default=1.0)
    args = p.parse_args()

    from transformers.utils import logging as _hf
    _hf.set_verbosity_error(); _hf.disable_progress_bar()
    device = "cuda:0"
    torch.cuda.set_device(0)
    place = nested_placement if args.placement == "nested" else interleaved_placement
    d = deadline_s()

    rec = {"policy": args.policy, "placement": args.placement, "clip_id": args.clip_id,
           "deadline_s": d, "release_after": args.release_after,
           "calls": [], "error": None}

    try:
        config = load_config("/root/oom-free-alpamayo/r1_config.json")
        adapter = get_adapter(config.model.kind)

        class A: pass
        a = A()
        a.model_id = None; a.model_cache_dir = None; a.model_revision = None
        a.attn_implementation = None; a.local_files_only = False
        a.num_traj_samples = adapter.default_num_traj_samples
        a.max_generation_length = adapter.default_max_generation_length

        start_free = free_gb()
        chosen = select_profile(start_free, safety_gb=args.safety_gb)
        if chosen is None:
            raise RuntimeError(f"no profile fits {start_free:.2f}GB at startup")
        rec["startup_profile"] = chosen.name
        rec["startup_free_gb"] = start_free

        loaded = adapter.load(a, config)
        adapter.setup_essentials(loaded, device)
        n_vlm = len(loaded.vlm_layers)
        resident = set(place(chosen.k, n_vlm))
        for i in resident:
            loaded.vlm_layers[i].to(device)
        torch.cuda.synchronize()
        pipe = TriHookPipeline(loaded.vlm_layers, loaded.vit_blocks,
                                loaded.expert_layers, list(resident), device=device)
        inputs = prepare_inputs_for_clip(loaded, args.clip_id, device)
        current = chosen

        pipe.start_iteration()          # warmup, untimed
        with torch.no_grad():
            adapter.run(loaded, inputs, a)
        torch.cuda.synchronize()

        # Held memory is reachable by us even though it is not reported free,
        # so it must be counted or the policy would never upgrade.
        held = current.footprint_gb

        if args.marker:
            args.marker.write_text(str(time.time()))

        for i in range(args.n_calls):
            transition_s = 0.0
            switched_from = None

            if args.policy == "adaptive":
                cand = select_profile(free_gb(), already_held_gb=held,
                                      safety_gb=args.safety_gb)
                if cand is not None and cand.k > current.k:
                    switched_from = current.name
                    t0 = time.perf_counter()
                    target = set(place(cand.k, n_vlm))
                    pipe.remove()
                    for j in target - resident:
                        loaded.vlm_layers[j].to(device)
                    for j in resident - target:
                        loaded.vlm_layers[j].to("cpu")
                    torch.cuda.synchronize()
                    pipe = TriHookPipeline(loaded.vlm_layers, loaded.vit_blocks,
                                            loaded.expert_layers, list(target),
                                            device=device)
                    transition_s = time.perf_counter() - t0
                    resident, current, held = target, cand, cand.footprint_gb

            t0 = time.perf_counter()
            pipe.start_iteration()
            with torch.no_grad():
                adapter.run(loaded, inputs, a)
            torch.cuda.synchronize()
            latency = time.perf_counter() - t0

            # The transition is charged to the call that paid for it, because
            # that is what a deadline actually experiences.
            total = latency + transition_s
            rec["calls"].append({
                "i": i, "profile": current.name, "k": current.k,
                "inference_s": latency, "transition_s": transition_s,
                "total_s": total, "switched_from": switched_from,
                "met_deadline": total <= d,
            })
        pipe.remove()

    except Exception as e:
        rec["error"] = f"{type(e).__name__}: {str(e)[:300]}"

    args.output.write_text(json.dumps(rec, indent=2))

    calls = rec["calls"]
    if calls:
        misses = sum(1 for c in calls if not c["met_deadline"])
        switches = sum(1 for c in calls if c["switched_from"])
        tot_trans = sum(c["transition_s"] for c in calls)
        mean_inf = sum(c["inference_s"] for c in calls) / len(calls)
        print(f"[{args.policy:<8} {args.placement:<8}] switches={switches} "
              f"transition_total={tot_trans:.2f}s mean_inference={mean_inf:.2f}s "
              f"misses={misses}/{len(calls)}")
    else:
        print(f"[{args.policy:<8} {args.placement:<8}] FAILED {rec['error']}")


if __name__ == "__main__":
    main()
