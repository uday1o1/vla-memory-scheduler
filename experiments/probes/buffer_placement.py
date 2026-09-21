"""Does where the staging buffers land decide the slow state?

Rebuilding the streaming pipeline, while moving no layers and changing no
residency, leaves inference about thirteen percent slower for a stretch, and
does so on roughly one rebuild in five. Residency change is therefore not the
cause. Reconstruction is, and reconstruction does one thing that matters to
the hot path: it allocates two staging buffers on the device, each sized for
the largest streamed layer, through which every prefetched layer passes.

Those allocations are the obvious suspect, because their addresses are the
only part of the rebuilt state that is not determined by the configuration. If
some placements are slower to stream through than others, a rebuild is a draw
from that distribution and the outcome persists because the buffers persist.

This records the address and alignment of both buffers on every rebuild
alongside the latency that follows, so the two can be compared directly. A
correlation identifies the mechanism and points at a fix, since buffers can be
allocated once and reused across residency changes rather than reallocated.
No correlation rules the addresses out and leaves reconstruction doing
something else.
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
MB = 1024 * 1024


def buffer_facts(pipe) -> list[dict]:
    """Address, size and alignment of each staging buffer this pipeline holds."""
    facts = []
    seen = set()
    for attr in ("hooks", "modules", "_hooks"):
        for h in (getattr(pipe, attr, None) or []):
            for b in (getattr(h, "gpu_bufs", None) or []):
                if b is None or id(b) in seen:
                    continue
                seen.add(id(b))
                ptr = b.data_ptr()
                facts.append({
                    "ptr": ptr,
                    "ptr_hex": hex(ptr),
                    "bytes": b.numel() * b.element_size(),
                    # Largest power of two that divides the address, capped,
                    # which is what "how aligned is it" means in practice.
                    "align_kb": min(1 << 20, (ptr & -ptr)) // 1024,
                    "offset_in_2mb": ptr % (2 * MB),
                })
    return facts


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--k", type=int, default=16)
    p.add_argument("--rebuilds", type=int, default=14)
    p.add_argument("--per-block", type=int, default=5)
    p.add_argument("--output", type=Path, default=RESULTS / "buffer_placement.json")
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

    resident = nested_placement(args.k, n_vlm)
    for i in resident:
        loaded.vlm_layers[i].to(device)
    torch.cuda.synchronize()

    def build():
        return TriHookPipeline(loaded.vlm_layers, loaded.vit_blocks,
                               loaded.expert_layers, list(resident), device=device)

    def run_once(pipe) -> float:
        t0 = time.perf_counter()
        pipe.start_iteration()
        with torch.no_grad():
            adapter.run(loaded, inputs, a)
        torch.cuda.synchronize()
        return time.perf_counter() - t0

    results = {"meta": run_metadata(args), "k": args.k, "rebuilds": []}
    print(f"\nK={args.k}, residency never changes, only the pipeline is rebuilt\n")
    print(f"{'#':<4}{'median s':<12}{'buffers':<10}{'first ptr':<20}"
          f"{'align KB':<11}{'offset in 2MB'}")

    for r in range(args.rebuilds):
        pipe = build()
        facts = buffer_facts(pipe)
        for _ in range(3):
            run_once(pipe)
        times = [run_once(pipe) for _ in range(args.per_block)]
        med = statistics.median(times)
        results["rebuilds"].append({
            "rebuild": r + 1, "median": med, "times": times, "buffers": facts,
        })
        f0 = facts[0] if facts else {}
        print(f"{r + 1:<4}{med:<12.3f}{len(facts):<10}"
              f"{f0.get('ptr_hex', 'n/a'):<20}{f0.get('align_kb', 0):<11}"
              f"{f0.get('offset_in_2mb', 0)}")
        pipe.remove(); del pipe; gc.collect()

    args.output.write_text(json.dumps(results, indent=2))

    meds = [e["median"] for e in results["rebuilds"]]
    lo, hi = min(meds), max(meds)
    mid = (lo + hi) / 2
    fast = [e for e in results["rebuilds"] if e["median"] < mid]
    slow = [e for e in results["rebuilds"] if e["median"] >= mid]

    print(f"\n=== {len(fast)} fast, {len(slow)} slow, spread "
          f"{100 * (hi - lo) / lo:.2f}% ===")
    if not slow or not fast or not results["rebuilds"][0]["buffers"]:
        print("  Only one cluster appeared, or no buffers were visible, so the")
        print("  addresses cannot be compared against the outcome this time.")
    else:
        def summarize(group, label):
            aligns = [b["align_kb"] for e in group for b in e["buffers"]]
            offs = [b["offset_in_2mb"] for e in group for b in e["buffers"]]
            print(f"  {label:<6} median latency {statistics.median(e['median'] for e in group):.3f}s")
            print(f"         alignment KB      {sorted(set(aligns))}")
            print(f"         offset in 2MB     {sorted(set(offs))[:6]}")
        summarize(fast, "fast")
        summarize(slow, "slow")
        fa = {b["align_kb"] for e in fast for b in e["buffers"]}
        sa = {b["align_kb"] for e in slow for b in e["buffers"]}
        print()
        if fa & sa:
            print("  The two groups share buffer alignments, so alignment alone does")
            print("  not decide the outcome.")
        else:
            print("  The groups do not share a buffer alignment, so where the staging")
            print("  buffers land is what separates a fast rebuild from a slow one.")
    print(f"\nSaved -> {args.output}")


if __name__ == "__main__":
    main()
