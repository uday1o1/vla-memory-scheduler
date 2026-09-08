# Project Proposal, CMPE 249, Fall 2026

**Title:** Deadline and Contention Aware GPU Memory Scheduling for Vision-Language-Action Models
**Team:** Uday Arora
**Track:** Deployment Track

## 1. Problem Formulation

**Domain problem.** NVIDIA's Alpamayo is an open 10-billion-parameter Vision-Language-Action model for autonomous driving. Its weights total 22GB, exceeding the 12 to 16GB VRAM typical of automotive and consumer GPUs. A peer-reviewed system, `oom-free-alpamayo` (IEEE RTCSA 2026), addresses this by streaming model layers between CPU and GPU memory on demand, with an offline profiling pass deciding which layers stay permanently GPU-resident to maximize throughput. This fits the model in memory, at reported inference times of 4.09 seconds (RTX 5070 Ti) and 15.46 seconds (RTX 3080 Ti), both configurations that fail without it. Alpamayo's own paper states that autonomous driving requires approximately 100ms end-to-end latency for real-time operation, a gap of 40 to 150 times. The residency policy that makes the model fit has no concept of a deadline and no awareness of contention from a second workload sharing the GPU. Five related systems were reviewed in full text (see `literature-survey.md`, sources 1 through 5); none combine deadline awareness, contention awareness, and intra-model memory-residency decisions for a VLA model. The closest, published in June 2026, is itself runtime-adaptive, but for a different problem: it load-balances a model that already fits across CPU and GPU, reacting to raw utilization and queue signals to migrate a small, fixed set of boundary layers. This project instead fits a model that does not fit in GPU memory at all, reacting to an explicit deadline budget and sensed GPU contention across the full residency decision, not a bounded migration zone.

**Scope of the claim.** This project does not aim to make Alpamayo meet the 100ms target. Even the newest published competitor, with substantially more engineering investment, reaches only 306 to 408ms on real driving VLA models. The contribution is relative: whether a deadline and contention aware residency policy measurably reduces the deadline-miss rate and degrades more gracefully under contention than the existing static policy, not whether it achieves real-time inference outright.

**Input and output.** Input: Alpamayo's normal multi-camera and trajectory-history input, plus two additional signals not used by the current system: a per-call deadline budget, and a live GPU contention estimate obtained via NVML or DCGM, or a known injected level during controlled experiments. Output: the model's normal trajectory and reasoning output, plus a per-call record of which layers were kept resident and the resulting latency, for analysis.

**Target metrics and success criteria, fixed in advance.**

Primary metric: deadline-miss rate, the fraction of inference calls exceeding a stated latency budget, compared between the adaptive policy and the unmodified static baseline under injected GPU contention. The budget is defined relative to the static baseline's own achievable latency per scenario, since the underlying 100ms figure is not reachable by either system; a miss is defined as a call meaningfully worse than the system's own achievable floor, fixed before data collection.

Secondary metric: the shape of the latency degradation curve as contention increases, testing whether the adaptive policy produces smaller latency variance than the static policy, which has no mechanism to react to contention at all.

Statistical test: a paired Wilcoxon signed-rank test, with effect size reported as matched-pairs rank-biserial correlation, at least eight repetitions per scenario, alpha of 0.05.

Power caveat, stated in advance: a single model family and a small sample size mean only large effects are reliably detectable; this is disclosed before data collection, not after a null result.

Contention scenarios: a steady compute-bound load using a synthetic contention generator modeled on Kutukcu et al. (literature-survey.md source 6), and a bursty load injected by a second process at randomized intervals, covering both sustained and intermittent contention shapes. Both run as an independent process sharing the GPU through CUDA MPS, which allows two independent processes to share one GPU without a hardware repartition, matching the deployment scenario of an unmodified second workload appearing at runtime.

Null-result framing, fixed in advance: if the adaptive policy shows no statistically detected improvement, this is reported as exactly that, a valid and reportable outcome given that this combination has not been studied before.

## 2. Proposed Technical Approach

**Foundation.** The project forks `aveeslab/oom-free-alpamayo`, whose codebase is small and cleanly modular, with a documented adapter interface and a single function implementing the GPU-Resident Layer Decision Policy. The swapping mechanism itself, Sequential and Pipelined Demand Layering, is correct existing infrastructure and is not modified. Early implementation verifies that this function's isolation holds in practice and that CUDA MPS supports two concurrent processes on the target GPU, since MPS is not officially supported inside every container platform.

1. Baseline: the unmodified upstream static residency policy, run exactly as released.
2. Adaptive policy: replaces the one-time offline profiling decision with a policy that re-evaluates residency at runtime using the remaining deadline budget and a live GPU-occupancy signal. Under low contention and ample budget, it behaves like the static baseline, which is already close to optimal in that regime. Under high contention or a tight budget, it prioritizes layers that most reduce worst-case latency variance, even where that is not the global throughput optimum.
3. Contention injection: a synthetic dummy-workload contention generator and a bursty-arrival generator, run as an independent process sharing the GPU through CUDA MPS.
4. Evaluation: both policies run under both contention scenarios, at least eight repetitions each, on the same hardware, model, and inputs, differing only in the residency policy.

**Data sources.** The model, `nvidia/Alpamayo-R1-10B`, is open and ungated, licensed under OpenMDW-1.1, an open academic license permitting free non-commercial use. The codebase is MIT licensed. Neither requires special access. Representative driving-scene input frames come from either NVIDIA's `PhysicalAI-Autonomous-Vehicles` dataset, which is gated by an automatic license agreement rather than manual review, or `Jonas-a11y/alpamayo-carla-bridge`, a CARLA-simulator integration for Alpamayo maintained independently of the `oom-free-alpamayo` authors, which avoids the licensed dataset entirely. Only a small sample of driving frames is needed; the full dataset is not required.

**Technical modifications.** A runtime-adaptive layer-residency policy replacing the static one; a deadline and contention sensing layer feeding that policy; a contention-injection harness; and an evaluation pipeline comparing deadline-miss rate and degradation-curve shape rather than raw throughput alone.

## 3. Device Available and Maintainer

Device available: NVIDIA RTX 5050 (desktop) for code development; model loading, inference, and contention experiments run on a rented A100-class GPU, since Alpamayo-R1-10B's 22GB footprint exceeds the local card's memory. Maintainer: Uday Arora; Claude Code access is requested to support implementation throughout the semester.
