"""What does demoting a layer cost, and does clean eviction remove it?

Shedding resident layers has been measured at about twenty times the cost of
restoring the same number, 5.71s against 0.28s for eight layers on the RTX
3090. The upstream pipeline demotes a layer by copying it to host memory and
then pinning it, and pinning allocates page-locked memory. Inference never
writes the weights, so a pinned host copy kept from the start would make the
device copy redundant and demotion a matter of dropping it.

This measures both mechanisms on the same cycle of residency changes,
covering demotion and promotion at two sizes each. The first cycle is
discarded so that no measured transition carries one-time costs. After every
transition one inference call is timed and its trajectory kept, so that the
outputs of the two mechanisms, and of every residency level, can be compared
element by element: a faster mechanism that changed the output would not be
an optimization.

Run once per mechanism, in both orders, since processes are compared:
    python experiments/placement/eviction.py --demotion copy
    python experiments/placement/eviction.py --demotion clean
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch  # noqa: E402

from scheduler.paths import R1_CONFIG, RESULTS, bootstrap  # noqa: E402

bootstrap()

from alpamayo_memopt import load_config  # noqa: E402
from alpamayo_memopt.models import TriHookPipeline, get_adapter  # noqa: E402
from scheduler.inputs import prepare_inputs_for_clip  # noqa: E402
from scheduler.placement import nested_placement, upstream_placement  # noqa: E402
from scheduler.provenance import run_metadata  # noqa: E402
from scheduler.residency import Residency  # noqa: E402

CLIP = "d497f01b-4f68-4c27-9a6c-55872a1d6bd6"
CYCLE = (33, 24, 16, 24, 33, 16)  # demote 9, 8, restore 8, 9, demote 17, then 16->33 restores 17


def extract_traj(out):
    if isinstance(out, torch.Tensor) and out.is_floating_point() and out.numel() > 1:
        return out.detach().float().cpu()
    if isinstance(out, dict):
        out = list(out.values())
    if isinstance(out, (list, tuple)):
        for v in out:
            r = extract_traj(v)
            if r is not None:
                return r
    return None


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--demotion", choices=["copy", "clean"], required=True)
    p.add_argument("--placement", choices=["nested", "upstream"], default="nested")
    p.add_argument("--cycles", type=int, default=4, help="measured cycles, after one discarded")
    p.add_argument("--output", type=Path, default=None)
    args = p.parse_args()
    out = args.output or RESULTS / f"eviction_{args.demotion}_{args.placement}.json"

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
    inputs = prepare_inputs_for_clip(loaded, CLIP, device)
    place = nested_placement if args.placement == "nested" else upstream_placement

    def build(resident):
        return TriHookPipeline(loaded.vlm_layers, loaded.vit_blocks,
                               loaded.expert_layers, resident, device=device)

    t0 = time.perf_counter()
    res = Residency(loaded.vlm_layers, place, build, device, demotion=args.demotion)
    setup_s = time.perf_counter() - t0
    host_copy_gb = sum(t.numel() * t.element_size() for m in res.master.values()
                       for d in m.values() for t in d.values()) / 1024 ** 3
    print(f"{args.demotion}: setup {setup_s:.1f}s, host copies held {host_copy_gb:.2f}GB")

    def infer():  # the upstream adapter seeds its own sampling on every call
        t = time.perf_counter()
        res.pipe.start_iteration()
        with torch.no_grad():
            o = adapter.run(loaded, inputs, a)
        torch.cuda.synchronize()
        return time.perf_counter() - t, extract_traj(o)

    res.apply(CYCLE[-1])
    infer()  # warm
    record = {"meta": run_metadata(args), "setup_s": setup_s,
              "host_copy_gb": host_copy_gb, "transitions": [], "trajectories": {}}
    print(f"\n{'cycle':<7}{'K':<10}{'promoted':<10}{'demoted':<9}{'seconds':<10}{'next call s'}")
    for c in range(args.cycles + 1):
        prev = res.resident and len(res.resident)
        for k in CYCLE:
            cost = res.apply(k)
            call_s, traj = infer()
            cost.update(cycle=c, from_k=prev, call_s=call_s,
                        reserved_gb=torch.cuda.memory_reserved() / 1024 ** 3)
            prev = k
            if c == 0:
                continue  # discarded
            record["transitions"].append(cost)
            if traj is not None and str(k) not in record["trajectories"]:
                record["trajectories"][str(k)] = traj.flatten().tolist()
            print(f"{c:<7}{cost['from_k']:>2}->{k:<6}{cost['promoted']:<10}{cost['demoted']:<9}"
                  f"{cost['total_s']:<10.3f}{call_s:.3f}")
    res.close()

    # Bit-identity across residency levels within this process.
    trajs = {k: torch.tensor(v) for k, v in record["trajectories"].items()}
    ref = next(iter(trajs.values()))
    record["max_abs_diff_across_k"] = max(float((t - ref).abs().max()) for t in trajs.values())
    out.write_text(json.dumps(record))

    print("\n=== Median transition seconds by kind ===")
    kinds = {}
    for t in record["transitions"]:
        kinds.setdefault((t["promoted"], t["demoted"]), []).append(t["total_s"])
    for (pr, de), v in sorted(kinds.items()):
        print(f"  promote {pr:<3} demote {de:<3} {statistics.median(v):.3f}s  (n={len(v)})")
    print(f"\n  outputs across residency levels: max abs diff "
          f"{record['max_abs_diff_across_k']:.3e}")
    other = out.with_name(out.name.replace(
        args.demotion, "clean" if args.demotion == "copy" else "copy"))
    if other.exists():
        theirs = json.loads(other.read_text())["trajectories"]
        shared = sorted(set(theirs) & set(record["trajectories"]), key=int)
        diffs = {k: float((torch.tensor(theirs[k]) - torch.tensor(record["trajectories"][k]))
                          .abs().max()) for k in shared}
        print(f"  outputs against the other mechanism ({other.name}): "
              + ", ".join(f"K={k} {v:.3e}" for k, v in diffs.items()))
    print(f"Saved -> {out}")


if __name__ == "__main__":
    main()
