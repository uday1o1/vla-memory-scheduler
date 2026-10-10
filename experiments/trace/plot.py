"""Figures for the trace experiment, and an animation of one run for the demo.

    python experiments/trace/plot.py                    # figures/trace_*.png
    python experiments/trace/plot.py --animate out.mp4  # needs ffmpeg

The timeline shows one repetition call by call: the tenant's demand, the
residency each arm chose, and each call's latency with the residency change
charged to it, against the deadline. The three arms shown are the deployed
system (evenly spaced placement, copy demotion), an Oneiros-style controller
(evenly spaced placement, host copy kept) and nested placement with a host
copy kept.
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
SHOWN = {
    ("runtime", "upstream", "copy"): ("Deployed: evenly spaced, copy back", "#b5452f"),
    ("runtime", "upstream", "clean"): ("Oneiros-style: evenly spaced, host copy", "#d9a520"),
    ("runtime", "nested", "clean"): ("Nested, host copy", "#2f6db5"),
}


def arm(r):
    a = r["args"]
    return (a["policy"], a["placement"], a["demotion"]), int(a["phase_calls"])


def timeline(runs, phase_calls, rep, ax_top, ax_bot, upto=None):
    sel = {arm(r)[0]: r for r in runs
           if arm(r)[1] == phase_calls and r["rep"] == rep and arm(r)[0] in SHOWN}
    any_run = next(iter(sel.values()))
    n = len(any_run["demands"]) if upto is None else upto
    x = range(n)
    ax_top.step(x, any_run["demands"][:n], where="mid", color="#555", lw=1.5)
    ax_top.set_ylabel("tenant GB")
    ax_top.set_ylim(-0.5, max(any_run["demands"]) + 1.5)
    for key, (label, color) in SHOWN.items():
        r = sel.get(key)
        if r is None:
            continue
        calls = r["calls"][:n]
        ys = [c["total_s"] if c["total_s"] is not None else float("nan") for c in calls]
        ax_bot.plot(range(len(ys)), ys, marker="o", ms=3, lw=1.2, color=color, label=label)
    d = any_run["deadline_s"]
    ax_bot.axhline(d, color="#222", ls="--", lw=1)
    ax_bot.text(0, d, f" deadline {d:.2f}s", va="bottom", fontsize=8)
    ax_bot.set_xlabel("call")
    ax_bot.set_ylabel("latency incl. change, s")
    ax_bot.set_xlim(-0.5, len(any_run["demands"]) - 0.5)
    ax_bot.legend(fontsize=7, loc="upper right")


def bars(runs, ax):
    acc = defaultdict(list)
    for r in runs:
        key, L = arm(r)
        if key[0] in ("runtime", "deadline"):
            d = r["deadline_s"]
            acc[(key, L)].append(sum(1 for c in r["calls"] if c["inference_s"] is not None
                                     and c["inference_s"] <= d < c["total_s"]))
    keys = sorted({k for k, _ in acc})
    churns = sorted({L for _, L in acc}, reverse=True)
    w = 0.8 / len(churns)
    for j, L in enumerate(churns):
        ys = [statistics.mean(acc[(k, L)]) if acc.get((k, L)) else 0 for k in keys]
        ax.bar([i + j * w for i in range(len(keys))], ys, width=w,
               label=f"{L} calls per phase")
    ax.set_xticks([i + w * (len(churns) - 1) / 2 for i in range(len(keys))])
    ax.set_xticklabels([f"{p}\n{pl}\n{de}" for p, pl, de in keys], fontsize=7)
    ax.set_ylabel("misses caused by a residency change\n(mean per 48-call run)")
    ax.legend(fontsize=8)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, default=REPO / "data" / "trace_results.json")
    p.add_argument("--outdir", type=Path, default=REPO / "figures")
    p.add_argument("--phase-calls", type=int, default=2)
    p.add_argument("--rep", type=int, default=0)
    p.add_argument("--animate", type=Path, default=None)
    args = p.parse_args()
    runs = json.loads(args.input.read_text())["runs"]
    args.outdir.mkdir(exist_ok=True)

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(9, 5), sharex=True,
                                 gridspec_kw={"height_ratios": [1, 3]})
    timeline(runs, args.phase_calls, args.rep, a1, a2)
    fig.tight_layout()
    fig.savefig(args.outdir / f"trace_timeline_L{args.phase_calls}.png", dpi=160)

    fig, ax = plt.subplots(figsize=(9, 4))
    bars(runs, ax)
    fig.tight_layout()
    fig.savefig(args.outdir / "trace_transition_misses.png", dpi=160)
    print(f"Saved figures to {args.outdir}")

    if args.animate:
        from matplotlib.animation import FFMpegWriter
        fig, (a1, a2) = plt.subplots(2, 1, figsize=(9, 5), sharex=True,
                                     gridspec_kw={"height_ratios": [1, 3]})
        n = len(runs[0]["demands"])
        writer = FFMpegWriter(fps=4)
        with writer.saving(fig, str(args.animate), dpi=120):
            for upto in range(1, n + 1):
                a1.clear(); a2.clear()
                timeline(runs, args.phase_calls, args.rep, a1, a2, upto=upto)
                ymax = max((c["total_s"] or 0) for r in runs for c in r["calls"]
                           if arm(r)[0] in SHOWN)
                a2.set_ylim(0, ymax * 1.1)
                fig.tight_layout()
                writer.grab_frame()
        print(f"Saved animation to {args.animate}")


if __name__ == "__main__":
    main()
