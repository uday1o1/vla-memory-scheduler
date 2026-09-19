"""VRAM accounting and a competing allocator used to create memory pressure."""
from __future__ import annotations

import os
import time

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch  # noqa: E402


def gb(x: float) -> float:
    return x / (1024 ** 3)


def hog_vram(hold_gb: float, duration_s: float, ready_evt, device_idx: int = 0):
    """Allocate and hold `hold_gb` of VRAM, for use as a separate process."""
    torch.cuda.set_device(device_idx)
    n_elems = int(hold_gb * (1024 ** 3) / 4)
    try:
        block = torch.empty(n_elems, dtype=torch.float32, device=f"cuda:{device_idx}")
        block.fill_(1.0)
        torch.cuda.synchronize()
        ready_evt.set()
        time.sleep(duration_s)
    except Exception as e:
        print(f"    [hog] FAILED to allocate {hold_gb}GB: {type(e).__name__}: {str(e)[:120]}")
        ready_evt.set()
