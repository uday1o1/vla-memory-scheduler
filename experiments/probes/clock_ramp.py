"""Is the measurement-order penalty caused by GPU clocks ramping?

On a host that caps the card below its rated power and denies clock locking,
the arm measured first in an A/B latency comparison runs about 2.4 seconds
slower, and three warmup passes do not absorb it. Clock ramp is the natural
explanation: the first block of work runs while the card is still climbing out
of its idle state, and the second runs at sustained clocks. That explanation
has been asserted but not checked.

This samples the SM clock and power draw in a background thread while running
the same two-arm comparison, so the clocks during each measurement block can
be read off directly. If the first block runs at a markedly lower clock than
the second, the explanation holds. If the clocks are the same, it does not and
the penalty has another cause.

Run on the machine that exhibits the penalty. A card that is not power capped
is expected to show flat clocks and no penalty, which is the control.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import threading
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


class ClockSampler(threading.Thread):
    """Samples SM clock, power and throttle state until told to stop."""

    def __init__(self, interval: float = 0.25):
        super().__init__(daemon=True)
        self.interval = interval
        self.samples: list[tuple[float, int, float]] = []
        self._stop = threading.Event()

    def run(self) -> None:
        query = "clocks.sm,power.draw"
        while not self._stop.is_set():
            try:
                out = subprocess.run(
                    ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=5,
                )
                if out.returncode == 0:
                    sm, pw = out.stdout.strip().splitlines()[0].split(",")
                    self.samples.append((time.perf_counter(), int(sm), float(pw)))
            except Exception:
                pass
            self._stop.wait(self.interval)

    def stop(self) -> None:
        self._stop.set()

    def between(self, t0: float, t1: float) -> list[tuple[float, int, float]]:
        return [s for s in self.samples if t0 <= s[0] <= t1]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--reps", type=int, default=5)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--nested-first", action="store_true")
    p.add_argument("--preheat", type=int, default=0,
                   help="placement changes to perform before measuring, to "
                        "reproduce the accumulated state of a long benchmark run")
    p.add_argument("--output", type=Path, default=RESULTS / "clock_ramp.json")
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

    def switch(indices):
        nonlocal current
        t = set(indices)
        for i in t - current:
            loaded.vlm_layers[i].to(device)
        for i in current - t:
            loaded.vlm_layers[i].to("cpu")
        torch.cuda.synchronize()
        current = t
        return TriHookPipeline(loaded.vlm_layers, loaded.vit_blocks,
                               loaded.expert_layers, list(t), device=device)

    order = [("upstream", interleaved_placement), ("nested", nested_placement)]
    if args.nested_first:
        order.reverse()

    if args.preheat:
        # A long run leaves the allocator holding blocks from dozens of prior
        # placements. If the penalty needs that state rather than measurement
        # order, reproducing the churn should reproduce the penalty.
        print(f"Preheating with {args.preheat} placement changes...")
        cycle = [7, 14, 10, 33, 16, 24]
        for n in range(args.preheat):
            k = cycle[n % len(cycle)]
            rule = nested_placement if n % 2 else interleaved_placement
            ph = switch(rule(k, n_vlm))
            ph.start_iteration()
            with torch.no_grad():
                adapter.run(loaded, inputs, a)
            torch.cuda.synchronize()
            ph.remove()
        print("  preheat done")

    sampler = ClockSampler()
    sampler.start()
    time.sleep(2.0)
    idle = sampler.samples[-4:]
    print(f"\nIdle before any work: sm={[s[1] for s in idle]} MHz, "
          f"power={[round(s[2]) for s in idle]} W")

    results = {"meta": run_metadata(args), "k": args.k, "blocks": {}}
    print(f"\n{'position':<10}{'rule':<12}{'latency s':<13}{'sm MHz mean':<14}"
          f"{'sm MHz min':<13}{'power W mean'}")
    for pos, (name, rule) in enumerate(order, start=1):
        pipe = switch(rule(args.k, n_vlm))
        for _ in range(args.warmup):
            pipe.start_iteration()
            with torch.no_grad():
                adapter.run(loaded, inputs, a)
            torch.cuda.synchronize()

        ts = []
        block_start = time.perf_counter()
        for _ in range(args.reps):
            t0 = time.perf_counter()
            pipe.start_iteration()
            with torch.no_grad():
                adapter.run(loaded, inputs, a)
            torch.cuda.synchronize()
            ts.append(time.perf_counter() - t0)
        block_end = time.perf_counter()
        pipe.remove()

        sm = [s[1] for s in sampler.between(block_start, block_end)]
        pw = [s[2] for s in sampler.between(block_start, block_end)]
        mean_lat = statistics.mean(ts)
        results["blocks"][name] = {
            "position": pos, "latency_mean": mean_lat, "latencies": ts,
            "sm_mhz": sm, "power_w": pw,
        }
        print(f"{pos:<10}{name:<12}{mean_lat:<13.3f}"
              f"{statistics.mean(sm) if sm else 0:<14.0f}"
              f"{min(sm) if sm else 0:<13}{statistics.mean(pw) if pw else 0:.0f}")

    sampler.stop()
    args.output.write_text(json.dumps(results, indent=2))

    first, second = order[0][0], order[1][0]
    b1, b2 = results["blocks"][first], results["blocks"][second]
    lat_gap = 100 * (b1["latency_mean"] - b2["latency_mean"]) / b2["latency_mean"]
    if b1["sm_mhz"] and b2["sm_mhz"]:
        clk_gap = 100 * (statistics.mean(b1["sm_mhz"]) - statistics.mean(b2["sm_mhz"])) \
            / statistics.mean(b2["sm_mhz"])
    else:
        clk_gap = float("nan")

    print(f"\n=== First block ({first}) against second block ({second}) ===")
    print(f"  latency  {lat_gap:+.2f}%   (positive means the first block was slower)")
    print(f"  SM clock {clk_gap:+.2f}%   (negative means the first block ran slower clocks)")
    print()
    if lat_gap > 3 and clk_gap < -3:
        print("  The first block was slower and ran at lower clocks: consistent with ramp.")
    elif lat_gap > 3:
        print("  The first block was slower but clocks were comparable, so ramp does")
        print("  NOT explain it and the cause is elsewhere.")
    else:
        print("  No penalty on this run, so this machine does not exhibit the effect.")
    print(f"\nSaved -> {args.output}")


if __name__ == "__main__":
    main()
