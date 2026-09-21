"""Does this machine hold one inference speed, or flip between two?

Several effects measured on one RTX 3090 were attributed to residency
transitions: that arriving at a level by shedding costs about thirteen
percent, that the cost does not decay, and that it does not scale with pinned
allocations. The last measurement undermined all of them. The same transition,
from K=30 to K=16, measured twice in one session, gave no penalty the first
time and 12.06 percent the second.

Two measurements of one transition cannot differ by that much if the
transition is what sets the speed. The alternative is that the machine settles
into one of two speeds for stretches at a time, and that every experiment so
far has been reading which state it happened to be in rather than the effect
of whatever was being varied.

This changes nothing and measures repeatedly. One residency level, one clip,
no transitions, no placement changes: blocks of calls separated by short
pauses, each block summarized on its own. A machine with one speed produces
one cluster. A machine with two produces two, and then the attributions made
earlier are artifacts of sampling whichever state was current.
"""
from __future__ import annotations

import argparse
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
    p.add_argument("--k", type=int, default=16)
    p.add_argument("--blocks", type=int, default=10)
    p.add_argument("--per-block", type=int, default=6)
    p.add_argument("--pause", type=float, default=5.0,
                   help="idle seconds between blocks, since a state that "
                        "survives idling is different from one that does not")
    p.add_argument("--output", type=Path, default=RESULTS / "bistability.json")
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

    # Placed once. Nothing below changes residency.
    resident = nested_placement(args.k, n_vlm)
    for i in resident:
        loaded.vlm_layers[i].to(device)
    torch.cuda.synchronize()
    pipe = TriHookPipeline(loaded.vlm_layers, loaded.vit_blocks,
                           loaded.expert_layers, list(resident), device=device)

    def run_once() -> float:
        t0 = time.perf_counter()
        pipe.start_iteration()
        with torch.no_grad():
            adapter.run(loaded, inputs, a)
        torch.cuda.synchronize()
        return time.perf_counter() - t0

    for _ in range(3):
        run_once()

    results = {"meta": run_metadata(args), "k": args.k, "blocks": []}
    print(f"\nK={args.k}, no residency changes at any point\n")
    print(f"{'block':<8}{'median s':<12}{'min':<10}{'max':<10}{'spread'}")
    for b in range(args.blocks):
        times = [run_once() for _ in range(args.per_block)]
        med = statistics.median(times)
        results["blocks"].append({"block": b + 1, "times": times, "median": med})
        print(f"{b + 1:<8}{med:<12.3f}{min(times):<10.3f}{max(times):<10.3f}"
              f"{max(times) - min(times):.3f}")
        if args.pause:
            time.sleep(args.pause)

    args.output.write_text(json.dumps(results, indent=2))

    meds = [blk["median"] for blk in results["blocks"]]
    lo, hi = min(meds), max(meds)
    spread_pct = 100 * (hi - lo) / lo
    mid = (lo + hi) / 2
    low_group = [m for m in meds if m < mid]
    high_group = [m for m in meds if m >= mid]

    print(f"\n=== Across {len(meds)} blocks, nothing changed between them ===")
    print(f"  fastest block median {lo:.3f}s")
    print(f"  slowest block median {hi:.3f}s")
    print(f"  spread {spread_pct:.2f}%")
    print(f"  blocks below the midpoint {len(low_group)}, at or above {len(high_group)}")
    if low_group and high_group:
        print(f"  low cluster  mean {statistics.mean(low_group):.3f}s")
        print(f"  high cluster mean {statistics.mean(high_group):.3f}s")

    print()
    if spread_pct > 5 and len(low_group) >= 2 and len(high_group) >= 2:
        print("  Two clusters with nothing varied between them. The machine holds")
        print("  more than one speed, so effects attributed to transitions on this")
        print("  machine were reading which state it was in.")
    elif spread_pct > 5:
        print("  Block medians vary by more than five percent with nothing varied,")
        print("  so measurements here are unreliable at that scale whatever the")
        print("  shape of the variation.")
    else:
        print("  One cluster. The machine holds a single speed, so the variation")
        print("  seen earlier came from what was being changed, not from drift.")
    print(f"\nSaved -> {args.output}")


if __name__ == "__main__":
    main()
