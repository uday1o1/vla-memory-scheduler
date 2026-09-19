"""One co-located VLA tenant. Launched as an independent OS process.

Waits until a shared wall-clock start time so multiple tenants genuinely
overlap, then runs N inference calls at a fixed residency K and writes its
own latency record.
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--k", type=int, required=True)
    p.add_argument("--n-calls", type=int, default=20)
    p.add_argument("--start-at", type=float, required=True, help="unix timestamp")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--clip-id", default="d497f01b-4f68-4c27-9a6c-55872a1d6bd6")
    p.add_argument("--tenant-id", default="t0")
    args = p.parse_args()

    from transformers.utils import logging as _hf
    _hf.set_verbosity_error(); _hf.disable_progress_bar()
    device = "cuda:0"
    torch.cuda.set_device(0)

    record = {"tenant_id": args.tenant_id, "k": args.k, "clip_id": args.clip_id,
              "load_ok": False, "latencies": [], "error": None}

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
        n_vlm = len(loaded.vlm_layers)
        resident = set(interleaved_placement(args.k, n_vlm)) if args.k > 0 else set()
        for i in resident:
            loaded.vlm_layers[i].to(device)
        torch.cuda.synchronize()
        pipe = TriHookPipeline(loaded.vlm_layers, loaded.vit_blocks,
                                loaded.expert_layers, list(resident), device=device)
        inputs = prepare_inputs_for_clip(loaded, args.clip_id, device)

        free_after_load, total = torch.cuda.mem_get_info()
        record["load_ok"] = True
        record["process_used_gb"] = (total - free_after_load) / (1024 ** 3)

        # warmup (not timed)
        pipe.start_iteration()
        with torch.no_grad():
            adapter.run(loaded, inputs, a)
        torch.cuda.synchronize()

        # Sync: all tenants begin timed calls together.
        while time.time() < args.start_at:
            time.sleep(0.01)

        for _ in range(args.n_calls):
            t0 = time.perf_counter()
            pipe.start_iteration()
            with torch.no_grad():
                adapter.run(loaded, inputs, a)
            torch.cuda.synchronize()
            record["latencies"].append(time.perf_counter() - t0)

        pipe.remove()
    except Exception as e:
        record["error"] = f"{type(e).__name__}: {str(e)[:300]}"

    args.output.write_text(json.dumps(record, indent=2))
    status = "OK" if record["load_ok"] and not record["error"] else f"FAILED ({record['error']})"
    n = len(record["latencies"])
    mean = sum(record["latencies"]) / n if n else float("nan")
    print(f"[{args.tenant_id}] K={args.k} {status} n={n} mean={mean:.3f}s")


if __name__ == "__main__":
    main()
