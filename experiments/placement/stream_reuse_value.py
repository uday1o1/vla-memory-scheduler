"""What does sharing the prefetch stream change for a policy that switches?

Reconstructing the streaming pipeline draws a new prefetch stream, and on one
machine that stream is intermittently slower, costing about thirteen percent
for as long as the pipeline lives. Carrying the stream across reconstructions
removes it. That is established by repeating reconstruction many times at one
setting.

What follows for a policy is a separate question, and the obvious way to ask
it fails. Crossing the setting with something else and running each cell once
cannot resolve an effect that fires on about a quarter of reconstructions: an
arm's total turns on whether it happened to fire, which is enough to reverse
two arms and did.

So this repeats whole trials instead. Each trial is a short run that switches
residency a few times, exactly as a policy would, and the two settings are
compared as distributions over trials rather than as two numbers. What matters
for a deployment is not only the average but how often a trial comes out slow,
since that is what a deadline sees.
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
from scheduler.inputs import prepare_inputs_for_clip  # noqa: E402
from scheduler.placement import nested_placement  # noqa: E402
from scheduler.provenance import run_metadata  # noqa: E402

CLIP = "d497f01b-4f68-4c27-9a6c-55872a1d6bd6"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--k-a", type=int, default=24)
    p.add_argument("--k-b", type=int, default=16)
    p.add_argument("--trials", type=int, default=8, help="trials per setting")
    p.add_argument("--switches", type=int, default=3, help="switches within a trial")
    p.add_argument("--calls", type=int, default=3, help="calls after each switch")
    p.add_argument("--slow-threshold", type=float, default=5.0,
                   help="percent above the setting's own best trial at which a "
                        "trial counts as having hit the slow state")
    p.add_argument("--output", type=Path, default=RESULTS / "stream_reuse_value.json")
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

    def place(k: int):
        nonlocal current
        t = set(nested_placement(k, n_vlm))
        for i in t - current:
            loaded.vlm_layers[i].to(device)
        for i in current - t:
            loaded.vlm_layers[i].to("cpu")
        torch.cuda.synchronize()
        current = t
        return TriHookPipeline(loaded.vlm_layers, loaded.vit_blocks,
                               loaded.expert_layers, list(t), device=device)

    def run_once(pipe) -> float:
        t0 = time.perf_counter()
        pipe.start_iteration()
        with torch.no_grad():
            adapter.run(loaded, inputs, a)
        torch.cuda.synchronize()
        return time.perf_counter() - t0

    def trial(reuse: bool) -> dict:
        carried, calls, pipe = None, [], None
        for sw in range(args.switches + 1):
            k = args.k_a if sw % 2 == 0 else args.k_b
            if pipe is not None:
                pipe.remove(); del pipe; gc.collect()
            pipe = place(k)
            if reuse and carried is not None:
                for attr, saved in carried.items():
                    h = getattr(pipe, attr, None)
                    if h is not None and saved is not None:
                        h.set_bufs(list(h.gpu_bufs), None, saved)
            if reuse and carried is None:
                carried = {}
                for attr in ("vlm_hook", "vis_hook", "exp_hook"):
                    h = getattr(pipe, attr, None)
                    carried[attr] = getattr(h, "prefetch_stream", None) if h else None
            run_once(pipe)  # settle
            # Only the level both settings share is compared, so the mixture of
            # levels within a trial cannot differ between settings by chance.
            if k == args.k_b:
                calls.extend(run_once(pipe) for _ in range(args.calls))
            else:
                for _ in range(args.calls):
                    run_once(pipe)
        pipe.remove(); del pipe; gc.collect()
        return {"median_call_s": statistics.median(calls), "calls": calls}

    # Discarded trial so neither setting is the first to run.
    print("Warm-up trial, discarded...")
    trial(False)

    results = {"meta": run_metadata(args), "settings": {}}
    for reuse in (False, True):
        name = "reuse_stream" if reuse else "rebuild_stream"
        print(f"\n=== {name}, {args.trials} trials ===")
        meds = []
        for t in range(args.trials):
            r = trial(reuse)
            meds.append(r["median_call_s"])
            print(f"  trial {t + 1:<3}median call {r['median_call_s']:.3f}s")
        results["settings"][name] = {"trial_medians": meds}

    args.output.write_text(json.dumps(results, indent=2))

    print(f"\n=== Trials at K={args.k_b}, compared as distributions ===")
    print(f"{'setting':<18}{'best':<10}{'median':<10}{'worst':<10}{'spread':<10}{'slow trials'}")
    for name, blk in results["settings"].items():
        m = blk["trial_medians"]
        best = min(m)
        slow = [x for x in m if 100 * (x - best) / best > args.slow_threshold]
        blk["slow_trials"] = len(slow)
        print(f"{name:<18}{best:<10.3f}{statistics.median(m):<10.3f}{max(m):<10.3f}"
              f"{100 * (max(m) - best) / best:<10.2f}{len(slow)}/{len(m)}")

    rb = results["settings"]["rebuild_stream"]
    ru = results["settings"]["reuse_stream"]
    print()
    if rb["slow_trials"] > ru["slow_trials"]:
        print(f"  Sharing the stream removed the slow trials, {rb['slow_trials']} of "
              f"{args.trials} down to {ru['slow_trials']}. A policy that switches pays "
              f"the penalty at the rate on the left and need not.")
    elif rb["slow_trials"] == ru["slow_trials"] == 0:
        print("  Neither setting produced a slow trial, so this run says nothing "
              "about the remedy. The hazard is intermittent and may not have fired.")
    else:
        print("  Sharing the stream did not reduce the slow trials here, which "
              "contradicts the reconstruction measurement and needs explaining "
              "before the remedy is relied on.")
    print(f"\nSaved -> {args.output}")


if __name__ == "__main__":
    main()
