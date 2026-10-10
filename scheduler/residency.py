"""Changing which layers are GPU-resident, and what that costs.

Two choices govern the cost of a residency change. Placement decides how many
layers move; this module takes the placement as a function. Demotion decides
what moving a layer off the device involves, and that is what this module
adds.

  copy   what the upstream pipeline does. A demoted layer is copied back to
         host memory and then pinned, because streaming needs a pinned source.
         Pinning allocates page-locked memory, which is slow, and this is why
         shedding a layer costs several times more than restoring one.

  clean  keep a pinned host copy of every layer for the life of the process.
         Inference never writes the weights, so the device copy of a resident
         layer is always identical to the host copy, and demoting it needs no
         transfer at all: the parameters are pointed back at the host copy and
         the device memory is released. This is clean eviction, as for an
         unmodified page in a memory hierarchy. It costs host memory: every
         resident layer is held twice, once on each side.

Both modes rebuild the streaming pipeline after a change, as the upstream
design requires, and both release freed device memory to the driver, since a
co-located process can only allocate what this process has given back.
"""
from __future__ import annotations

import gc
import time

import torch


class Residency:
    def __init__(self, layers, place, build, device: str, demotion: str = "copy",
                 total: int | None = None):
        """`build(resident)` constructs a streaming pipeline for that resident
        set. The pipeline is owned here so that no caller can keep an old one,
        and its device buffers, alive across a rebuild."""
        if demotion not in ("copy", "clean"):
            raise ValueError(f"unknown demotion mode {demotion!r}")
        self.layers, self.place, self.build = layers, place, build
        self.device, self.demotion = device, demotion
        self.pipe = None
        self.total = total if total is not None else len(layers)
        self.resident: set[int] = set()
        self.master: dict[int, dict[str, dict[str, torch.Tensor]]] = {}
        if demotion == "clean":
            # One-time cost, paid before anything is timed.
            for i, ly in enumerate(layers):
                self.master[i] = {
                    "p": {n: self._pinned(p.data) for n, p in ly.named_parameters()},
                    "b": {n: self._pinned(b.data) for n, b in ly.named_buffers()},
                }
                self._point_at_host(i)

    @staticmethod
    def _pinned(t: torch.Tensor) -> torch.Tensor:
        t = t.detach().to("cpu").contiguous()
        return t if t.is_pinned() else t.pin_memory()

    def _point_at_host(self, i: int) -> None:
        ly, m = self.layers[i], self.master[i]
        for n, p in ly.named_parameters():
            p.data = m["p"][n]
        for n, b in ly.named_buffers():
            b.data = m["b"][n]

    def close(self) -> None:
        if self.pipe is not None:
            self.pipe.remove()
            self.pipe = None
        gc.collect()

    def _promote(self, i: int) -> None:
        if self.demotion == "clean":
            ly, m = self.layers[i], self.master[i]
            for n, p in ly.named_parameters():
                p.data = m["p"][n].to(self.device, non_blocking=True)
            for n, b in ly.named_buffers():
                b.data = m["b"][n].to(self.device, non_blocking=True)
        else:
            self.layers[i].to(self.device)

    def _demote(self, i: int) -> None:
        if self.demotion == "clean":
            self._point_at_host(i)
        else:
            self.layers[i].to("cpu")  # the pipeline pins it on rebuild

    def apply(self, k: int) -> dict:
        """Move to residency k, rebuild the pipeline, and return the cost.

        Timing covers everything a deadline would see: removing the old
        pipeline, moving layers, releasing memory, and rebuilding.
        """
        target = set(self.place(k, self.total))
        promote, demote = target - self.resident, self.resident - target
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        if self.pipe is not None:
            self.pipe.remove()
            self.pipe = None
        gc.collect()
        for i in sorted(promote):
            self._promote(i)
        for i in sorted(demote):
            self._demote(i)
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        t_move = time.perf_counter() - t0
        self.resident = target
        self.pipe = self.build(sorted(target))
        torch.cuda.synchronize()
        total = time.perf_counter() - t0
        return {"k": k, "promoted": len(promote), "demoted": len(demote),
                     "move_s": t_move, "total_s": total}
