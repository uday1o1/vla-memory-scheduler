# Project Proposal, CMPE 249, Fall 2026

**Title:** Residency Scheduling for Oversized Vision-Language-Action Models: Profile Switching Under GPU Contention
**Team:** Uday Arora
**Track:** Deployment Track

This revision responds to feedback on the submitted proposal: reduce initial scope, establish feasibility first, start with a simple policy selecting among a few pre-profiled residency configurations rather than optimizing layer placement on every call, and evaluate the overhead of changing residency itself.

## 1. Problem Formulation

**Domain problem.** NVIDIA's Alpamayo is an open 10-billion-parameter Vision-Language-Action model for autonomous driving. Its weights total 22GB, exceeding the 12 to 16GB VRAM typical of automotive and consumer GPUs. A peer-reviewed system, `oom-free-alpamayo` (IEEE RTCSA 2026), addresses this by streaming model layers between CPU and GPU memory on demand, with an offline profiling pass deciding which layers stay permanently GPU-resident. This fits the model in memory, at reported inference times of 4.09 seconds (RTX 5070 Ti) and 15.46 seconds (RTX 3080 Ti), both configurations that fail without it.

The residency count is a tunable knob: more resident layers is faster but occupies more VRAM. The existing system fixes that knob at one point, chosen by capacity arithmetic on an idle GPU, and never revisits it. Two consequences follow. The tradeoff curve is unpublished, so the cost of operating at any other point is unknown. And because the choice is made against an idle GPU, it does not hold when the GPU is not idle.

**Scope of the claim.** This project does not aim to reach the approximately 100ms real-time target stated in Alpamayo's own paper; the newest published competitor reaches 306 to 408ms. The question is narrower and answerable: does selecting among pre-profiled residency configurations using measured GPU state reduce the deadline-miss rate relative to the fixed offline choice, under a controlled competing workload, and does the cost of changing residency undermine that benefit.

**Input and output.** Input: Alpamayo's normal multi-camera and trajectory-history input, plus one signal the current system does not use, the free GPU memory reported by `torch.cuda.mem_get_info`, which is visible across processes and requires no elevated privilege. Output: the model's normal trajectory and reasoning output, plus a per-call record of the residency profile in force and the resulting latency.

**Target metrics and success criteria, fixed in advance.**

Primary metric: deadline-miss rate, the fraction of inference calls exceeding the stated budget, compared between the fixed static residency and profile switching under a controlled competing GPU workload.

Reported alongside it: completion rate. The two arms can fail differently. A run that never produces a trajectory misses every deadline, but that is a different failure from producing trajectories too slowly, and collapsing them would conceal which is occurring.

Secondary metric: the cost of changing residency, measured separately by direction, since offloading and restoring are not symmetric.

Deadline definition: the latency of the middle profile times 1.15. The anchor matters. An earlier version of this design anchored the deadline to the fastest configuration that fits, which made the comparison uninformative, because every other configuration then misses by construction and both arms behave identically. Anchoring to the middle profile leaves two profiles meeting the deadline and one missing it, so the comparison can distinguish them. The policy module asserts this separation holds.

Statistical test: paired Wilcoxon signed-rank with matched-pairs rank-biserial effect size, eight repetitions per condition, alpha 0.05. Where every paired difference is zero the test is not reported, since a test on identical arms carries no information; that case is stated directly instead.

Power caveat, stated in advance: one model family and a small sample mean only large effects are reliably detectable.

Null-result framing, fixed in advance: a regime where switching does not help is reported as exactly that. Establishing where a mechanism does not apply bounds the claim.

**Contention scenarios.** Two forms of pressure, acting on different resources:

1. Memory pressure, an independent process holding GPU memory before the model process starts. This is the primary scenario, because residency governs memory and this is where the decision changes the outcome.
2. Compute contention, a duty-cycled saturating workload in an independent process sharing the GPU through CUDA MPS, modeled on Kutukcu et al. Reported as a boundary result.

## 2. Proposed Technical Approach

**Foundation.** The project forks `aveeslab/oom-free-alpamayo`. The swapping mechanism, Sequential and Pipelined Demand Layering, is correct existing infrastructure and is not modified. The residency decision is the surface under study.

**Milestone 1, feasibility and characterization.** Reproduce the system on the rented GPU and characterize latency and memory behavior across residency levels under a controlled competing workload, before implementing any policy.

**Milestone 2, profile switching.** Three pre-profiled configurations taken from the measured curves rather than chosen arbitrarily. The switching rule reads free GPU memory and selects the fastest profile that fits with margin. This replaces an earlier design that re-derived an arbitrary residency count on every inference call; both the feedback on the submitted proposal and this project's own transition-cost measurements argue against that approach, since changing residency costs a substantial fraction of an inference and an unconstrained policy can spend much of its budget reconfiguring.

**Milestone 3, evaluation.** Static residency against profile switching across memory pressure levels, eight repetitions each, reporting deadline-miss rate, completion rate, and latency.

**Milestone 4, residency change overhead.** Measure what changing residency costs, separated by direction, and determine whether that cost undermines the benefit of switching. This addresses the concern that a theoretically better placement may increase deadline misses if migration dominates inference.

**Stretch goal.** A finer-grained online scheduler, and co-location of multiple model instances at reduced residency.

**Data sources.** `nvidia/Alpamayo-R1-10B`, open and ungated under OpenMDW-1.1. Driving-scene inputs from NVIDIA's `PhysicalAI-Autonomous-Vehicles` dataset, which is gated and additionally requires an access token with the public-gated-repository permission enabled. Six distinct clips are used, cycled across the eight repetitions per condition, so input varies across repetitions while hardware is held constant. Both arms of a pair see the same clip, so the paired test compares like with like.

## 3. Target Environment Constraints

Confirmed by direct test on the rented GPU. They constrain what can be measured and are stated because they shape the metric definitions above.

- GPU graphics clock locking is denied by the hypervisor even with root and passwordless sudo, so timing runs cannot pin clocks. Clock frequency is logged instead, so throttling is detectable and disclosable.
- The DCGM Profiling module, which provides fine-grained SM-occupancy counters, fails to load. Hardware performance counter access is commonly restricted in multi-tenant GPU environments.
- `nvidia-smi` reports `utilization.gpu` as 100 percent at every duty cycle from 20 to 100 percent, so it detects whether any kernel is active rather than fractional occupancy and cannot calibrate contention intensity. Intensity is therefore defined as the duty cycle of the injected workload rather than measured.
- `torch.cuda.mem_get_info` is cross-process aware and requires no elevated privilege, so it is used as the memory-pressure signal and a deployed system would have the same signal available.

## 4. Device Available and Maintainer

Device available: NVIDIA RTX 5050 (desktop) for code development; model loading, inference, and contention experiments run on a rented RTX 3090 with 24GB VRAM, since Alpamayo-R1-10B's 22GB footprint exceeds the local card's memory. The submitted proposal named an A100-class GPU; the RTX 3090 was used instead, and is closer to the consumer and automotive class the problem targets.

Maintainer: Uday Arora; Claude Code access is requested to support implementation throughout the semester.
