"""What does running adaptively actually cost, with and without both fixes?

Two separate results bear on the cost of changing residency, and neither has
been measured against the other. Making the residency family nested cuts the
layers a change must move, by up to 21.84x at single-level adjustments.
Sharing the prefetch stream and events across reconstructions removes an
intermittent thirteen percent penalty that reconstruction otherwise leaves
behind. One acts on the transition, the other on everything after it.

An adaptive policy pays both, so the question is what they are worth together
on a workload that actually switches. The dynamic experiment measured one
large upgrade under a large change in available memory, where a gain above
fifty percent swamps either cost. The regime where they matter is the opposite
one: frequent modest adjustments, which is what a policy tracking a moving
memory signal produces.

Four arms cross the two fixes. Each alternates between two residency levels
for a fixed number of phases, rebuilding the pipeline on every change as the
upstream design requires, and every call and every transition is timed. The
comparison is total wall time to complete identical work, which is what a
deployment pays.

One limit of this design is worth knowing before reading its output. On a
machine where reconstruction intermittently leaves a slower pipeline, the
hazard fires on roughly a quarter of reconstructions, each arm performs only a
handful, and there is one sample per cell. An arm's total then turns largely
on whether the hazard happened to fire in it, which is enough to reverse the
ordering of two arms. Run on such a machine, this measures the placement rule,
whose effect is deterministic, and not the stream setting, whose effect is
intermittent. The stream setting is measured properly by repeating
reconstruction many times at one setting, which probes/buffer_placement.py
does.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import statistics
import sys
import time
from pathlib import Path
from pathlib import Path as _Path

sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch  # noqa: E402

from scheduler.paths import R1_CONFIG, RESULTS, bootstrap  # noqa: E402

bootstrap()

from alpamayo_memopt import load_config  # noqa: E402
from alpamayo_memopt.models import TriHookPipeline, get_adapter  # noqa: E402
from alpamayo_memopt.profiler import interleaved_placement  # noqa: E402
from scheduler.inputs import prepare_inputs_for_clip  # noqa: E402
from scheduler.placement import nested_placement  # noqa: E402
from scheduler.provenance import run_metadata  # noqa: E402

CLIP = "d497f01b-4f68-4c27-9a6c-55872a1d6bd6"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--k-a", type=int, default=24)
    p.add_argument("--k-b", type=int, default=16)
    p.add_argument("--phases", type=int, default=6,
                   help="residency changes per arm")
    p.add_argument("--calls-per-phase", type=int, default=4)
    p.add_argument("--output", type=Path, default=RESULTS / "adaptive_cost.json")
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

    print("Loading model...")
    loaded = adapter.load(a, config)
    adapter.setup_essentials(loaded, device)
    n_vlm = len(loaded.vlm_layers)
    inputs = prepare_inputs_for_clip(loaded, CLIP, device)
    current: set[int] = set()

    def build(indices):
        return TriHookPipeline(loaded.vlm_layers, loaded.vit_blocks,
                               loaded.expert_layers, list(indices), device=device)

    def snapshot(pipe):
        out = {}
        for attr in ("vlm_hook", "vis_hook", "exp_hook"):
            h = getattr(pipe, attr, None)
            if h is None:
                continue
            out[attr] = {
                "bufs": list(getattr(h, "gpu_bufs", None) or []),
                "events": list(getattr(h, "compute_done", None) or []),
                "stream": getattr(h, "prefetch_stream", None),
            }
        return out

    def run_once(pipe) -> float:
        t0 = time.perf_counter()
        pipe.start_iteration()
        with torch.no_grad():
            adapter.run(loaded, inputs, a)
        torch.cuda.synchronize()
        return time.perf_counter() - t0

    def arm(rule, reuse: bool) -> dict:
        nonlocal current
        current = set()
        for i in range(n_vlm):
            loaded.vlm_layers[i].to("cpu")
        torch.cuda.synchronize()

        carried, moves, trans, calls = None, 0, [], []
        pipe = None
        for ph in range(args.phases + 1):
            k = args.k_a if ph % 2 == 0 else args.k_b
            target = set(rule(k, n_vlm))

            if pipe is not None:
                pipe.remove(); del pipe; gc.collect()
            t0 = time.perf_counter()
            moves += len(target - current) + len(current - target)
            for i in target - current:
                loaded.vlm_layers[i].to(device)
            for i in current - target:
                loaded.vlm_layers[i].to("cpu")
            torch.cuda.synchronize()
            current = target
            pipe = build(target)
            if reuse and carried is not None:
                for attr, saved in carried.items():
                    h = getattr(pipe, attr, None)
                    if h is not None and saved["bufs"]:
                        h.set_bufs(saved["bufs"], saved["events"], saved["stream"])
            torch.cuda.synchronize()
            trans.append(time.perf_counter() - t0)
            if reuse and carried is None:
                carried = snapshot(pipe)

            run_once(pipe)  # settle, not counted
            for _ in range(args.calls_per_phase):
                calls.append(run_once(pipe))

        pipe.remove(); del pipe; gc.collect()
        return {
            "moves": moves,
            "transition_s": sum(trans),
            "transitions": trans,
            "inference_s": sum(calls),
            "calls": calls,
            "total_s": sum(trans) + sum(calls),
            "median_call_s": statistics.median(calls),
        }

    results = {"meta": run_metadata(args), "arms": {}}
    print(f"\nAlternating K={args.k_a} and K={args.k_b}, {args.phases} changes, "
          f"{args.calls_per_phase} calls each\n")

    # A discarded arm first. Whichever arm runs first pays one-time costs the
    # others do not: layers reach the device from a cold start and host memory
    # is pinned for the first time. Measured first, that inflates its
    # transition total by roughly three times and reads as a property of
    # whatever that arm was testing.
    print("Discarded warm-up arm, so no measured arm is the first to run...")
    arm(nested_placement, False)
    print("  warm-up done\n")
    print(f"{'placement':<11}{'stream':<10}{'moves':<8}{'transition s':<15}"
          f"{'inference s':<14}{'total s':<11}{'median call'}")

    for pname, rule in (("nested", nested_placement), ("upstream", interleaved_placement)):
        for reuse in (False, True):
            r = arm(rule, reuse)
            results["arms"][f"{pname}_{'reuse' if reuse else 'rebuild'}"] = r
            print(f"{pname:<11}{'reuse' if reuse else 'rebuild':<10}{r['moves']:<8}"
                  f"{r['transition_s']:<15.2f}{r['inference_s']:<14.2f}"
                  f"{r['total_s']:<11.2f}{r['median_call_s']:.3f}")

    args.output.write_text(json.dumps(results, indent=2))

    best = min(results["arms"].items(), key=lambda kv: kv[1]["total_s"])
    worst = max(results["arms"].items(), key=lambda kv: kv[1]["total_s"])
    print(f"\n=== Identical work, total wall time ===")
    for name, r in sorted(results["arms"].items(), key=lambda kv: kv[1]["total_s"]):
        print(f"  {name:<20}{r['total_s']:.2f}s")
    print(f"\n  best {best[0]} at {best[1]['total_s']:.2f}s, "
          f"worst {worst[0]} at {worst[1]['total_s']:.2f}s, "
          f"{100 * (worst[1]['total_s'] - best[1]['total_s']) / best[1]['total_s']:.1f}% apart")
    print("\n  The two fixes act on different parts of the bill. Nested placement")
    print("  reduces the transition column by moving fewer layers; sharing the")
    print("  stream reduces the inference column by not leaving a slower pipeline")
    print("  behind. Whether either matters depends on how often residency moves.")
    print(f"\nSaved -> {args.output}")


if __name__ == "__main__":
    main()
