"""Figures for the trace experiment, and an animation of one run for the demo.

    python experiments/trace/plot.py                    # figures/trace_*.png
    python experiments/trace/plot.py --animate out.mp4  # needs ffmpeg

The timeline shows one repetition call by call: the tenant's demand, the
residency each arm chose, and each call's latency with the residency change
charged to it, against the deadline. The three arms shown are the deployed
system (evenly spaced placement, copy demotion), an Oneiros-style controller
(evenly spaced placement, host copy kept) and nested placement with a host
copy kept. The bar chart counts misses caused by a residency change for every
switching arm, and the sensitivity figure scores the four runtime arms against
deadlines other than the one fixed before the run.
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

RUNTIME_ARMS = {
    ("runtime", "upstream", "copy"): ("evenly spaced, copy back (deployed)", "#b5452f"),
    ("runtime", "nested", "copy"): ("nested, copy back", "#e08a6e"),
    ("runtime", "upstream", "clean"): ("evenly spaced, host copy (Oneiros-style)", "#d9a520"),
    ("runtime", "nested", "clean"): ("nested, host copy", "#2f6db5"),
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


# The two-by-two first, deployed system to both corrections, then the policy
# that changes less often.
BAR_ORDER = [
    (("runtime", "upstream", "copy"), "deployed\nevenly spaced\ncopy back"),
    (("runtime", "nested", "copy"), "nested\ncopy back"),
    (("runtime", "upstream", "clean"), "Oneiros-style\nevenly spaced\nhost copy"),
    (("runtime", "nested", "clean"), "nested\nhost copy"),
    (("deadline", "upstream", "copy"), "deadline policy\nevenly spaced\ncopy back"),
    (("deadline", "nested", "copy"), "deadline policy\nnested\ncopy back"),
]


def bars(runs, ax):
    """Mean per arm, with every repetition drawn as a point over its bar."""
    acc = defaultdict(list)
    for r in runs:
        key, L = arm(r)
        d = r["deadline_s"]
        acc[(key, L)].append(sum(1 for c in r["calls"] if c["inference_s"] is not None
                                 and c["inference_s"] <= d < c["total_s"]))
    keys = [(k, lab) for k, lab in BAR_ORDER if any(kk == k for kk, _ in acc)]
    churns = sorted({L for k, L in acc if k[0] in ("runtime", "deadline")}, reverse=True)
    w = 0.8 / len(churns)
    for j, L in enumerate(churns):
        xs = [i + j * w for i in range(len(keys))]
        ys = [statistics.mean(acc[(k, L)]) if acc.get((k, L)) else 0 for k, _ in keys]
        ax.bar(xs, ys, width=w, color=["#9bb8d9", "#2f6db5"][j % 2],
               label=f"{L} calls per phase")
        for x, (k, _) in zip(xs, keys):
            pts = acc.get((k, L), [])
            jit = [x + w * 0.3 * ((i % 5) - 2) / 2 for i in range(len(pts))]
            ax.scatter(jit, pts, s=8, color="#222", zorder=3)
    ax.set_xticks([i + w * (len(churns) - 1) / 2 for i in range(len(keys))])
    ax.set_xticklabels([lab for _, lab in keys], fontsize=7)
    ax.set_ylabel("misses caused by a residency change\n(per 48-call run)")
    ax.legend(fontsize=8)


def sensitivity(runs, axes):
    """Deadline misses of the runtime arms against the deadline.

    The runtime policy never consults the deadline, so its recorded calls can
    be scored against any deadline after the fact. All four arms choose the
    same profiles, so the gaps between them are what residency changes cost.
    Total misses are plotted rather than those caused by a change, because a
    deadline tight enough for inference alone to miss would otherwise make the
    count fall as the deadline tightens. The dotted line is the deadline fixed
    before the run.
    """
    mid = runs[0]["deadline_s"] / 1.15
    mults = [1.0 + i * 0.005 for i in range(71)]
    churns = sorted({arm(r)[1] for r in runs if arm(r)[0][0] == "runtime"}, reverse=True)
    for ax, L in zip(axes, churns):
        for key, (label, color) in RUNTIME_ARMS.items():
            rs = [r for r in runs if arm(r) == (key, L)]
            if not rs:
                continue
            ys = [statistics.mean(sum(c["total_s"] is None or c["total_s"] > mid * m
                                      for c in r["calls"]) for r in rs) for m in mults]
            ax.plot(mults, ys, color=color, lw=1.5, label=label)
        ax.axvline(1.15, color="#222", ls=":", lw=1)
        ax.set_title(f"{L} calls per phase", fontsize=9)
        ax.set_xlabel("deadline, multiple of the mid profile's latency")
    axes[0].set_ylabel("deadline misses\n(mean per 48-call run)")
    axes[-1].legend(fontsize=7, loc="upper right")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, default=REPO / "data" / "trace_results_3090.json")
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

    n_l = len({arm(r)[1] for r in runs if arm(r)[0][0] == "runtime"})
    fig, axes = plt.subplots(1, n_l, figsize=(4.5 * n_l, 3.6), sharey=True, squeeze=False)
    sensitivity(runs, list(axes[0]))
    fig.tight_layout()
    fig.savefig(args.outdir / "trace_deadline_sensitivity.png", dpi=160)
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
