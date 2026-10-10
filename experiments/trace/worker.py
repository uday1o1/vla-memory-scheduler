"""One arm of the trace experiment: the driving model under a changing budget.

Runs as its own process beside the driver, which plays the co-located tenant.
At every call boundary the two hand over in a fixed order through small files
in a shared directory, so that each call sees the same demand in every arm:

  pre{i}   worker reached the boundary before call i
  ok{i}    driver has released whatever the tenant no longer needs
  post{i}  worker has finished any residency change for call i
  go{i}    driver has tried to give the tenant its demand; call i may start

The residency change is charged to the call that follows it, because that is
what the deadline experiences.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch  # noqa: E402

from scheduler.paths import R1_CONFIG, bootstrap  # noqa: E402

bootstrap()

from alpamayo_memopt import load_config  # noqa: E402
from alpamayo_memopt.models import TriHookPipeline, get_adapter  # noqa: E402
from scheduler.inputs import prepare_inputs_for_clip  # noqa: E402
from scheduler.placement import nested_placement, upstream_placement  # noqa: E402
from scheduler.profiles import deadline_s  # noqa: E402
from scheduler.residency import Residency  # noqa: E402
from scheduler.trace import POLICIES, budget_gb, choose, expand  # noqa: E402


def wait_for(path: Path, timeout_s: float = 900) -> str:
    t0 = time.time()
    while not path.exists():
        if time.time() - t0 > timeout_s:
            raise TimeoutError(f"no {path.name} from the driver")
        time.sleep(0.02)
    return path.read_text()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--policy", choices=POLICIES, required=True)
    p.add_argument("--placement", choices=["nested", "upstream"], default="nested")
    p.add_argument("--demotion", choices=["copy", "clean"], default="copy")
    p.add_argument("--phase-calls", type=int, default=6)
    p.add_argument("--clip-id", required=True)
    p.add_argument("--sync-dir", type=Path, required=True)
    p.add_argument("--safety-gb", type=float, default=1.0)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()

    from transformers.utils import logging as _hf
    _hf.set_verbosity_error(); _hf.disable_progress_bar()
    device = "cuda:0"
    torch.cuda.set_device(0)
    total_gb = torch.cuda.mem_get_info(0)[1] / 1024 ** 3
    demands = expand(args.phase_calls)
    budgets = [budget_gb(total_gb, d, args.safety_gb) for d in demands]
    d = deadline_s()
    rec = {"args": {k: str(v) for k, v in vars(args).items()}, "total_gb": total_gb,
           "deadline_s": d, "demands": demands, "calls": [], "error": None}

    try:
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
        inputs = prepare_inputs_for_clip(loaded, args.clip_id, device)
        place = nested_placement if args.placement == "nested" else upstream_placement

        def build(resident):
            return TriHookPipeline(loaded.vlm_layers, loaded.vit_blocks,
                                   loaded.expert_layers, resident, device=device)

        res = Residency(loaded.vlm_layers, place, build, device, demotion=args.demotion)
        cur = choose(args.policy, budgets[0], None, budgets)
        res.apply(cur.k)

        def infer() -> float:
            t = time.perf_counter()
            res.pipe.start_iteration()
            with torch.no_grad():
                adapter.run(loaded, inputs, a)
            torch.cuda.synchronize()
            return time.perf_counter() - t

        infer()  # warm, untimed, before the driver starts the trace
        (args.sync_dir / "loaded").write_text(json.dumps({"k": cur.k}))

        for i, (b, demand) in enumerate(zip(budgets, demands)):
            (args.sync_dir / f"pre{i}").write_text("")
            wait_for(args.sync_dir / f"ok{i}")
            nxt = choose(args.policy, b, cur, budgets)
            cost = {"promoted": 0, "demoted": 0, "total_s": 0.0}
            if nxt.k != cur.k:
                cost = res.apply(nxt.k)
            switched_from, cur = (cur.name if nxt.k != cur.k else None), nxt
            (args.sync_dir / f"post{i}").write_text(json.dumps({"k": cur.k}))
            tenant = json.loads(wait_for(args.sync_dir / f"go{i}"))

            try:
                inference_s, err = infer(), None
            except torch.cuda.OutOfMemoryError as e:
                inference_s, err = None, f"OOM: {str(e)[:160]}"
                torch.cuda.empty_cache()
            total = None if inference_s is None else inference_s + cost["total_s"]
            rec["calls"].append({
                "i": i, "demand_gb": demand, "budget_gb": b,
                "profile": cur.name, "k": cur.k, "switched_from": switched_from,
                "promoted": cost["promoted"], "demoted": cost["demoted"],
                "transition_s": cost["total_s"], "inference_s": inference_s,
                "total_s": total, "error": err,
                "met_deadline": total is not None and total <= d,
                "tenant_held_gb": tenant["held_gb"], "tenant_admitted": tenant["admitted"],
                "tenant_wait_s": tenant["wait_s"],
                "stream": getattr(res.pipe.vlm_hook.prefetch_stream, "cuda_stream", None),
            })
        res.close()
    except Exception as e:
        rec["error"] = f"{type(e).__name__}: {str(e)[:300]}"
    args.output.write_text(json.dumps(rec, indent=2))
    calls = rec["calls"]
    miss = sum(not c["met_deadline"] for c in calls)
    refused = sum(not c["tenant_admitted"] for c in calls)
    print(f"[{args.policy:<14} {args.placement:<8} {args.demotion:<5} L={args.phase_calls}] "
          f"calls={len(calls)} misses={miss} tenant_refused={refused} "
          f"transition_s={sum(c['transition_s'] for c in calls):.2f} error={rec['error']}")


if __name__ == "__main__":
    main()
