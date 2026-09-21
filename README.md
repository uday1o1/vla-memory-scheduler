# Residency Scheduling for Oversized Vision-Language-Action Models

**Course:** CMPE 249, Intelligent Autonomous Systems, Fall 2026, Prof. Kaikai Liu
**Team:** Uday Arora
**Track:** Deployment Track

## Abstract

Vision-Language-Action (VLA) models for autonomous driving, such as NVIDIA's open 10-billion-parameter Alpamayo, are too large to fit in the memory of commodity and automotive-grade GPUs. A peer-reviewed system, `oom-free-alpamayo` (IEEE RTCSA 2026), fixes the fit problem by streaming model layers between CPU and GPU on demand, but the specific set of layers it keeps GPU-resident is decided once, offline, by capacity arithmetic on an idle GPU, and never revisited. This project treats that residency count as a characterized runtime knob rather than a fixed offline choice: it measures what the knob costs in latency and buys in memory footprint, evaluates selecting among pre-profiled residency configurations against the fixed static choice under a controlled competing workload, and measures the overhead of changing residency itself.

That overhead turned out to be the project's principal result. The upstream placement rule is not nested across residency levels, so changing the resident count reshuffles layers that did not need to move. Making the residency family nested removes that cost, provably: a transition costs the symmetric difference of the two residency sets, which is at least the change in count, with equality exactly when one set contains the other. A nested rule built by recursive bisection meets the bound at every transition, produces bit-identical model outputs, and is measured at up to 21.84x faster than upstream at the single-level adjustments a switching policy makes. The gain belongs to nestedness rather than to that particular rule: four simpler nested orders were tested and all are equally move-optimal, with several achieving better layer spread. Full numbers and sources: `proposal.md` and `literature-survey.md`.

## Deliverables

- `literature-survey.md`: SOTA survey (Deliverable B)
- `proposal.md`: problem formulation, technical approach, device and maintainer (Deliverable C)
- `novelty-feasibility-audit.md`: AI novelty and feasibility audit (Deliverable D)

## Repository Layout

```
scheduler/                  importable library, shared by every experiment
  paths.py                  resolves the repo, the upstream clone, and the results directory
  placement.py              nested residency placement via recursive bisection
  profiles.py               pre-profiled residency configurations and the switching rule
  policy.py                 continuous-K residency policy, evaluated under contention
  contention.py             competing GPU workload generator
  inputs.py                 driving-clip input preparation
  memory.py                 VRAM accounting and a competing allocator
  provenance.py             hardware and invocation recorded into every result

experiments/
  characterize/             latency and footprint per residency level
  placement/                move-optimality, transition cost, output equivalence
  switching/                static residency against profile switching
  dynamic/                  whether upgrading residency repays its own cost
  contention/               compute-contention boundary and interaction
  colocation/               two tenants sharing one GPU
  probes/                   bounded investigations that settled a design question

data/                       measured results, one JSON per run, suffixed by GPU
setup/setup_instance.sh     one-command GPU instance setup
```

## Results

**Static residency against profile switching.** Eight repetitions per condition,
six clips cycled, paired Wilcoxon signed-rank. The deadline is the middle
profile's latency times 1.15, which is 14.16s. Memory pressure is an
independent process holding VRAM before the model process starts.

| pressure | static: completion / miss | switching: completion / miss | paired test |
|---|---|---|---|
| 0GB | 1.00 / 0.00 | 1.00 / 0.00 | arms identical, no test |
| 4GB | 1.00 / 0.00 | 1.00 / 0.00 | arms identical, no test |
| 7GB | **0.00** / 1.00 | **1.00** / **0.00** | W=0, p=0.0078, rank-biserial -1.000 |
| 10GB | **0.00** / 1.00 | **1.00** / 0.75 | p=0.50, not significant on miss rate |

At 7GB and 10GB the static configuration does not load at all, so it produces
no trajectory. Switching selects a smaller profile and completes every call.
Two qualifications belong with this result. At 10GB switching completes but
still misses three quarters of its deadlines, and the miss-rate difference
there is not significant, so the gain at that pressure is in completing at all
rather than in meeting deadlines. And p=0.0078 is the smallest value a
signed-rank test can return for eight pairs, so the 7GB result is a complete
separation whose precision is bounded by the sample size, not a marginal one.

Completion rate is reported beside miss rate because the two arms fail
differently: a run that never produces a trajectory misses every deadline, but
that is a different failure from producing trajectories too slowly.

**Does changing residency mid-run repay its own cost?** This is the feedback's
specific concern, that migration latency could make a better configuration
miss more deadlines. Memory is held so the process starts on the `compact`
profile, then released part way through the run, and the adaptive arm upgrades
while the fixed arm does not.

| clip | before upgrade | after upgrade | change |
|---|---|---|---|
| d497f01b | 16.95s | 6.99s | **-58.8%** |
| 441057af | 13.53s | 6.44s | -52.4% |
| 59aba96d | 16.25s | 6.88s | -57.7% |
| 8825b1fa | 14.34s | 6.60s | -54.0% |

The upgrade repays itself many times over: a gain above fifty percent against
a migration cost of a few seconds paid once. Both placement rules were run and
reach the same latency afterwards, 6.99s against 6.95s on the first clip,
which is the parity result again in a different setting. The qualification is
that this is a large profile change under a large change in available memory;
it does not establish that smaller or more frequent adjustments repay
themselves, and the rebuild cost described below is a reason to think frequent
ones may not.

**Compute contention, a negative result.** A duty-cycled saturating workload
sharing the GPU under CUDA MPS slows every residency level proportionally, by
0.988x, 1.005x and 1.031x at K=16, 24 and 33. Residency has no lever against
compute contention, which is why memory pressure is the primary scenario.

**Residency change overhead.** Changing residency costs 0.3 to 4.7 seconds
against 7 to 17 seconds of inference, and is asymmetric: shedding layers is
three to six times dearer than restoring them, because offloading pins host
memory while restoring is a plain device copy. The nested placement rule
reduces this, by up to 21.84x at the single-level adjustments a switching
policy makes.

## Reproducing

Every experiment is a standalone script run from the repository root. Paths are
resolved from each file's own location, so the checkout can live anywhere.

**Without a GPU.** The placement results are combinatorial and the statistics run
against the measurements committed in `data/`, so both reproduce on any machine:

```
pip install -r requirements.txt
python scheduler/placement.py              # nesting, spread and move counts
python experiments/placement/theory.py     # move-optimality and the ablation
python experiments/switching/analyze.py    # static against profile switching
python experiments/contention/analyze.py   # the compute-contention boundary
```

The two scripts that compare against the upstream rule report that it is
unavailable and skip that column when the Alpamayo packages are absent, rather
than failing.

**With a GPU.** Taking the measurements needs a CUDA GPU with at least 12GB of
VRAM, the upstream Alpamayo packages, and access to a gated dataset:

```
bash setup/setup_instance.sh     # clones upstream, installs, fetches weights, profiles
python experiments/characterize/calibrate.py     # latency per residency level
python experiments/characterize/vram_per_k.py    # footprint per residency level
python experiments/characterize/build_profiles.py
VLA_PROFILES=data/profiles.json \
  python experiments/switching/run.py            # the primary experiment
python experiments/placement/benchmark.py        # transition cost and latency parity
```

The profiles committed in `scheduler/profiles.py` were measured on an RTX 3090
and the deadline is derived from them, so they describe that card and not the
one running. `build_profiles.py` turns the two characterization runs into a
profile file for the machine in use, and refuses to emit one whose deadline
fails to separate the profiles, since a comparison against static residency
would then carry no information.

`nvidia/PhysicalAI-Autonomous-Vehicles` is gated and needs a Hugging Face token
whose "read public gated repos" permission is explicitly enabled, which is not
granted by default. Place it where `huggingface_hub` stores tokens, or export
`HF_TOKEN`. Setup honours `WORKDIR` for where the upstream clones live and
`VENV` for a virtualenv to activate; experiments honour `ALPAMAYO_HOME` for a
clone that is not a sibling of this repository and `VLA_RESULTS` for an output
directory other than `data/`.

Results are written with the GPU, driver, PCIe link and invocation that produced
them, so any number in `data/` can be traced to its hardware.

## Foundation

This project extends `aveeslab/oom-free-alpamayo`, the open-source release accompanying its IEEE RTCSA 2026 paper, which itself depends on NVIDIA's open Alpamayo model and inference source. Full citations and links are in `literature-survey.md`.
