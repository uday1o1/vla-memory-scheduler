# Residency Scheduling for Oversized Vision-Language-Action Models

**Course:** CMPE 249, Intelligent Autonomous Systems, Fall 2026, Prof. Kaikai Liu
**Team:** Uday Arora
**Track:** Deployment Track

## Abstract

Vision-Language-Action (VLA) models for autonomous driving, such as NVIDIA's open 10-billion-parameter Alpamayo, are too large to fit in the memory of commodity and automotive-grade GPUs. A peer-reviewed system, `oom-free-alpamayo` (IEEE RTCSA 2026), fixes the fit problem by streaming model layers between CPU and GPU on demand, but the specific set of layers it keeps GPU-resident is decided once, offline, by capacity arithmetic on an idle GPU, and never revisited. This project treats that residency count as a characterized runtime knob rather than a fixed offline choice: it measures what the knob costs in latency and buys in memory footprint, evaluates selecting among pre-profiled residency configurations against the fixed static choice under a controlled competing workload, and measures the overhead of changing residency itself.

That overhead turned out to be the project's principal result. The upstream placement rule is not nested across residency levels, so changing the resident count reshuffles layers that did not need to move. A nested rule built by recursive bisection is move-optimal at every transition, produces bit-identical model outputs, and matches upstream on both spread and steady-state latency across the residency levels the pre-profiled configurations use. Below those levels it is not free: spread parity fails at seven residency levels under K=19, and steady-state latency runs a mean of 1.028 percent slower at levels 7 to 14 on an RTX 5070 Ti. Full numbers and sources: `proposal.md` and `literature-survey.md`.

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
python experiments/placement/benchmark.py        # transition cost and latency parity
python experiments/switching/run.py              # the primary experiment
```

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
