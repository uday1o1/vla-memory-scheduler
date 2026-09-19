"""Calibrated GPU contention generator (Kutukcu et al. dummy-workload style).

Contention intensity is defined DIRECTLY as duty cycle (busy_ms / (busy_ms +
idle_ms)) of a saturating GEMM kernel, not measured via nvidia-smi or DCGM.

Why: nvidia-smi's utilization.gpu is a coarse "any kernel active during the
sample window" signal - it reports 100% at every duty cycle from 20% to 100%,
confirmed empirically (see novelty-feasibility-audit.md). DCGM's Profiling module
(the actual fine-grained SM-occupancy counters) fails to load in this
container - a common cloud-GPU restriction on hardware performance-counter
access for multi-tenant security. Neither path can measure fractional
utilization here.

Since a saturating GEMM kernel occupies ~100% of SM resources while running,
a 70% duty cycle directly constructs ~70% GPU occupancy without needing to
measure it - this is the locked "70% GPU_UTIL steady contention" target,
operationalized as duty cycle instead of a queried metric.

Bursty mode: the same steady generator gated on/off by a Poisson process
(lambda=0.2 inter-arrivals, mean 5s) with 500ms burst duration - matches the
locked experimental design.
"""
from __future__ import annotations

import argparse
import random
import time

import torch


def steady_load(stop_time: float, duty_cycle: float, device_idx: int = 0,
                 matrix_size: int = 4096, period_ms: float = 100.0):
    """Saturating GEMM loop, busy for `duty_cycle` fraction of each period_ms window."""
    torch.cuda.set_device(device_idx)
    device = f"cuda:{device_idx}"
    a = torch.randn(matrix_size, matrix_size, device=device, dtype=torch.float32)
    b = torch.randn(matrix_size, matrix_size, device=device, dtype=torch.float32)
    busy_s = (period_ms * duty_cycle) / 1000.0
    idle_s = (period_ms * (1 - duty_cycle)) / 1000.0
    while time.time() < stop_time:
        t_end = time.time() + busy_s
        while time.time() < t_end:
            c = a @ b
            torch.cuda.synchronize()  # per-iteration sync: kernel launches are
            # async, so without this the busy loop enqueues far more matmuls
            # than the GPU can finish within busy_s, and the eventual sync
            # blocks draining that backlog - blowing the intended duty cycle.
        if idle_s > 0:
            time.sleep(idle_s)


def bursty_load(total_duration_s: float, burst_duty_cycle: float = 1.0,
                 burst_duration_s: float = 0.5, lambda_rate: float = 0.2,
                 device_idx: int = 0, matrix_size: int = 4096, seed: int | None = None):
    """Poisson-arrival bursts (mean inter-arrival 1/lambda_rate seconds),
    each burst running `steady_load` at `burst_duty_cycle` for `burst_duration_s`.
    """
    rng = random.Random(seed)
    end_time = time.time() + total_duration_s
    while time.time() < end_time:
        inter_arrival_s = rng.expovariate(lambda_rate)
        time.sleep(min(inter_arrival_s, max(0.0, end_time - time.time())))
        if time.time() >= end_time:
            break
        burst_end = min(time.time() + burst_duration_s, end_time)
        steady_load(burst_end, burst_duty_cycle, device_idx, matrix_size)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["steady", "bursty"], default="steady")
    p.add_argument("--duty-cycle", type=float, default=0.70,
                   help="Steady mode: fraction of time busy (default 0.70, our locked target).")
    p.add_argument("--duration", type=float, default=60.0, help="seconds")
    p.add_argument("--burst-duration", type=float, default=0.5,
                   help="Bursty mode: seconds per burst (default 0.5, locked).")
    p.add_argument("--lambda-rate", type=float, default=0.2,
                   help="Bursty mode: Poisson rate, mean inter-arrival = 1/lambda (default 0.2 -> 5s mean).")
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--seed", type=int, default=None)
    args = p.parse_args()

    if args.mode == "steady":
        stop_time = time.time() + args.duration
        steady_load(stop_time, args.duty_cycle, args.device)
    else:
        bursty_load(args.duration, burst_duration_s=args.burst_duration,
                    lambda_rate=args.lambda_rate, device_idx=args.device, seed=args.seed)


if __name__ == "__main__":
    main()
