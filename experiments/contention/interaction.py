"""Does compute contention slow every residency level alike, or does it hurt
the streaming-heavy low levels more than the residency-heavy high ones?

If contention hurt low K more, a policy could protect latency by raising K
under contention; if it hurt high K more, by lowering it; if alike, residency
has no lever against compute contention. A second process runs a saturating
GEMM at a fixed duty cycle (scheduler/contention.py). Each repetition times
calls with and without it, alternating which comes first so that neither
condition always follows the other, and every call is kept.

The earlier measurement ran the two processes under CUDA MPS, so that their
kernels share the GPU concurrently rather than by time-slicing. Start it first
to match (as root: nvidia-cuda-mps-control -d); whether it was running is
recorded in the output.

    python experiments/contention/interaction.py --reps 5
"""
from __future__ import annotations

import argparse
import gc
import json
import multiprocessing as mp
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scheduler.paths import R1_CONFIG, RESULTS, bootstrap  # noqa: E402

bootstrap()

from alpamayo_memopt import load_config  # noqa: E402
from alpamayo_memopt.models import TriHookPipeline, get_adapter  # noqa: E402

from scheduler.inputs import prepare_inputs_for_clip  # noqa: E402
from scheduler.placement import upstream_placement  # noqa: E402
from scheduler.provenance import run_metadata  # noqa: E402


def dummy_compute_load(stop_flag, device_idx: int = 0, duty_cycle: float = 0.70):
    """Saturating GEMM in its own process at the given duty cycle."""
    from scheduler.contention import steady_load
    while not stop_flag.is_set():
        steady_load(time.time() + 1.0, duty_cycle=duty_cycle, device_idx=device_idx)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--k-values", type=int, nargs="+", default=[16, 24, 33])
    p.add_argument("--reps", type=int, default=5)
    p.add_argument("--calls", type=int, default=2, help="timed calls per condition per rep")
    p.add_argument("--duty-cycle", type=float, default=0.70)
    p.add_argument("--clip-id", default="d497f01b-4f68-4c27-9a6c-55872a1d6bd6")
    p.add_argument("--output", type=Path, default=RESULTS / "compute_interaction.json")
    args = p.parse_args()

    device = "cuda:0"
    torch.cuda.set_device(0)
    from transformers.utils import logging as _hf
    _hf.set_verbosity_error()
    _hf.disable_progress_bar()

    config = load_config(R1_CONFIG)
    adapter = get_adapter(config.model.kind)

    class Args:
        pass
    a = Args()
    a.model_id = None
    a.model_cache_dir = None
    a.model_revision = None
    a.attn_implementation = None
    a.local_files_only = False
    a.num_traj_samples = adapter.default_num_traj_samples
    a.max_generation_length = adapter.default_max_generation_length

    print("Loading model...")
    loaded = adapter.load(a, config)
    adapter.setup_essentials(loaded, device)
    inputs = prepare_inputs_for_clip(loaded, args.clip_id, device)
    n_vlm = len(loaded.vlm_layers)
    current: set[int] = set()
    mps = subprocess.run(["pgrep", "-f", "nvidia-cuda-mps-control"],
                         capture_output=True).returncode == 0
    record = {"meta": run_metadata(args), "mps": mps, "levels": {}}
    print(f"CUDA MPS {'running' if mps else 'not running'}")

    def contended(fn):
        stop = mp.Event()
        proc = mp.Process(target=dummy_compute_load, args=(stop, 0, args.duty_cycle))
        proc.start()
        time.sleep(1.0)  # let the load reach its duty cycle
        try:
            return fn()
        finally:
            stop.set()
            proc.join(timeout=5)
            if proc.is_alive():
                proc.terminate()

    for k in args.k_values:
        target = set(upstream_placement(k, n_vlm))
        for i in target - current:
            loaded.vlm_layers[i].to(device)
        for i in current - target:
            loaded.vlm_layers[i].to("cpu")
        current = target
        torch.cuda.synchronize()
        pipeline = TriHookPipeline(loaded.vlm_layers, loaded.vit_blocks,
                                   loaded.expert_layers, list(target), device=device)

        def run_once() -> float:
            t0 = time.perf_counter()
            pipeline.start_iteration()
            with torch.no_grad():
                adapter.run(loaded, inputs, a)
            torch.cuda.synchronize()
            return time.perf_counter() - t0

        def timed():
            return [run_once() for _ in range(args.calls)]

        run_once()  # warmup
        free, cont = [], []
        for rep in range(args.reps):
            if rep % 2 == 0:
                free += timed(); cont += contended(timed)
            else:
                cont += contended(timed); free += timed()
        ratio = statistics.median(cont) / statistics.median(free)
        record["levels"][str(k)] = {"no_contention_s": free, "contention_s": cont,
                                    "slowdown": ratio}
        print(f"K={k:<4}no contention {statistics.median(free):.3f}s  "
              f"contention {statistics.median(cont):.3f}s  slowdown {ratio:.3f}x  "
              f"(n={len(free)} each)")
        pipeline.remove()
        gc.collect()
        torch.cuda.empty_cache()
        args.output.write_text(json.dumps(record, indent=1))
    print(f"Saved -> {args.output}")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
