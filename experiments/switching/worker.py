"""One model process under a given policy, run against pre-existing memory pressure.

Started by experiments/switching/run.py AFTER the pressure holder has allocated, because
GPU memory is won by whoever allocates first: a competitor cannot take memory
this process already holds, so pressure only binds when it pre-exists.

static    always loads the upstream offline choice (K=33), which was profiled
          on an idle GPU and does not consult runtime memory
switching what the professor recommended: choose among pre-profiled
          configurations using measured free memory

The model is loaded once and all repetitions run inside this process. Each
repetition uses a different driving clip, so input varies while hardware and
load-time conditions are held constant. Reloading per repetition would add
load-to-load variance without adding signal.
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

from pathlib import Path as _Path  # noqa: E402
sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))

from ias.paths import R1_CONFIG, bootstrap  # noqa: E402

bootstrap()
from alpamayo_memopt import load_config
from alpamayo_memopt.models import TriHookPipeline, get_adapter
from alpamayo_memopt.profiler import interleaved_placement
from ias.inputs import prepare_inputs_for_clip
from ias.placement import nested_placement
from ias.profiles import deadline_s, free_gb, profile_for_k, select_profile


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--policy", choices=["static", "switching"], required=True)
    p.add_argument("--placement", choices=["nested", "upstream"], default="nested")
    p.add_argument("--reps", type=int, default=8)
    p.add_argument("--n-calls", type=int, default=20)
    p.add_argument("--clips", required=True, help="comma separated clip ids")
    p.add_argument("--pressure-gb", type=float, default=0.0)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--safety-gb", type=float, default=1.0)
    args = p.parse_args()

    clips = args.clips.split(",")

    from transformers.utils import logging as _hf
    _hf.set_verbosity_error(); _hf.disable_progress_bar()
    device = "cuda:0"
    torch.cuda.set_device(0)

    place = nested_placement if args.placement == "nested" else interleaved_placement
    d = deadline_s()

    rec = {
        "policy": args.policy, "placement": args.placement,
        "pressure_gb": args.pressure_gb, "deadline_s": d,
        "free_gb_at_start": None, "profile_chosen": None,
        "loaded": False, "error": None, "reps": [],
    }

    try:
        rec["free_gb_at_start"] = free_gb()

        if args.policy == "static":
            # The upstream offline choice, profiled on an idle GPU. It does not
            # consult runtime memory, which is exactly the behavior under test.
            chosen = profile_for_k(33)
        else:
            chosen = select_profile(rec["free_gb_at_start"], safety_gb=args.safety_gb)

        if chosen is None:
            rec["error"] = "no profile fits available memory"
            raise RuntimeError(rec["error"])

        rec["profile_chosen"] = {"name": chosen.name, "k": chosen.k,
                                  "expected_latency_s": chosen.latency_s,
                                  "footprint_gb": chosen.footprint_gb}

        config = load_config(R1_CONFIG)
        adapter = get_adapter(config.model.kind)

        class A: pass
        a = A()
        a.model_id = None; a.model_cache_dir = None; a.model_revision = None
        a.attn_implementation = None; a.local_files_only = False
        a.num_traj_samples = adapter.default_num_traj_samples
        a.max_generation_length = adapter.default_max_generation_length

        loaded = adapter.load(a, config)
        adapter.setup_essentials(loaded, device)
        n_vlm = len(loaded.vlm_layers)
        resident = set(place(chosen.k, n_vlm))
        for i in resident:
            loaded.vlm_layers[i].to(device)
        torch.cuda.synchronize()
        pipe = TriHookPipeline(loaded.vlm_layers, loaded.vit_blocks,
                                loaded.expert_layers, list(resident), device=device)
        rec["loaded"] = True

        for r in range(args.reps):
            clip = clips[r % len(clips)]
            inputs = prepare_inputs_for_clip(loaded, clip, device)

            pipe.start_iteration()          # warmup, untimed
            with torch.no_grad():
                adapter.run(loaded, inputs, a)
            torch.cuda.synchronize()

            lat = []
            for _ in range(args.n_calls):
                t0 = time.perf_counter()
                pipe.start_iteration()
                with torch.no_grad():
                    adapter.run(loaded, inputs, a)
                torch.cuda.synchronize()
                lat.append(time.perf_counter() - t0)

            misses = sum(1 for x in lat if x > d)
            rec["reps"].append({
                "rep": r, "clip_id": clip, "latencies": lat,
                "n_completed": len(lat), "misses_completed": misses,
                "misses_failed": args.n_calls - len(lat),
                "miss_rate": (misses + args.n_calls - len(lat)) / args.n_calls,
                "completion_rate": len(lat) / args.n_calls,
            })
        pipe.remove()

    except Exception as e:
        if not rec["error"]:
            rec["error"] = f"{type(e).__name__}: {str(e)[:300]}"
        # A run that never produced a trajectory is a total miss at every
        # repetition, not missing data: the system failed to deliver within
        # any budget. Recording it as absent would silently favor the arm
        # that crashed.
        for r in range(len(rec["reps"]), args.reps):
            rec["reps"].append({
                "rep": r, "clip_id": clips[r % len(clips)], "latencies": [],
                "n_completed": 0, "misses_completed": 0,
                "misses_failed": args.n_calls, "miss_rate": 1.0,
                "completion_rate": 0.0,
            })

    args.output.write_text(json.dumps(rec, indent=2))

    prof = rec["profile_chosen"]["name"] if rec["profile_chosen"] else "none"
    all_lat = [x for r in rec["reps"] for x in r["latencies"]]
    mean = sum(all_lat) / len(all_lat) if all_lat else float("nan")
    miss = sum(r["miss_rate"] for r in rec["reps"]) / len(rec["reps"]) if rec["reps"] else 1.0
    comp = sum(r["completion_rate"] for r in rec["reps"]) / len(rec["reps"]) if rec["reps"] else 0.0
    print(f"[{args.policy:<9} {args.pressure_gb:>4.1f}GB] profile={prof:<8} "
          f"completion={comp:.2f} miss_rate={miss:.2f} mean={mean:.2f}s"
          + (f"  ERROR={rec['error'][:70]}" if rec["error"] else ""))


if __name__ == "__main__":
    main()
