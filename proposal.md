# Project Proposal, CMPE 249, Fall 2026

**Title:** Characterizing the Latency and Memory Tradeoff in GPU Residency Scheduling for Oversized Vision-Language-Action Models
**Team:** Uday Arora
**Track:** Deployment Track

## 1. Problem Formulation

**Domain problem.** NVIDIA's Alpamayo is an open 10-billion-parameter Vision-Language-Action model for autonomous driving. Its weights total 22GB, exceeding the 12 to 16GB VRAM typical of automotive and consumer GPUs. A peer-reviewed system, `oom-free-alpamayo` (IEEE RTCSA 2026), addresses this by streaming model layers between CPU and GPU memory on demand, with an offline profiling pass deciding which layers stay permanently GPU-resident. This fits the model in memory, at reported inference times of 4.09 seconds (RTX 5070 Ti) and 15.46 seconds (RTX 3080 Ti), both configurations that fail without it.

That residency count is a tunable knob. Keeping more layers resident is faster but occupies more VRAM; keeping fewer is slower but leaves memory free. The existing system fixes this knob at a single point, chosen once by offline capacity arithmetic (VRAM budget divided by layer size) on an otherwise idle GPU, and never revisits it. The tradeoff curve that knob traces out has not been characterized, and the operating points other than maximum-fit have not been examined.

**What this project does.** It measures the full latency and VRAM footprint curve across residency levels, identifies operating points the fixed offline choice forecloses, and establishes where residency adaptation does and does not help. Five related systems were reviewed in full text (see `literature-survey.md`, sources 1 through 5); none characterize intra-model memory-residency as a runtime-tunable tradeoff for a VLA model. The closest, published in June 2026, is runtime-adaptive for a different problem: it load-balances a model that already fits across CPU and GPU, reacting to utilization and queue signals to migrate a bounded set of boundary layers. This project addresses a model that does not fit in GPU memory at all, and treats the residency count itself as the decision variable.

**Scope of the claim.** This project does not aim to make Alpamayo meet the approximately 100ms real-time target stated in Alpamayo's own paper. Even the newest published competitor reaches only 306 to 408ms on real driving VLA models. The contribution is characterization and boundary-setting: what the residency knob buys, what it costs, and which forms of resource pressure it can and cannot answer.

**Input and output.** Input: Alpamayo's normal multi-camera and trajectory-history input, plus one signal not used by the current system, the available GPU memory reported by `torch.cuda.mem_get_info`. Output: the model's normal trajectory and reasoning output, plus a per-call record of resident layer count, VRAM footprint, and latency.

**Target metrics and success criteria, fixed in advance.**

Primary metrics, reported as characterization rather than hypothesis tests:
1. Inference latency as a function of resident layer count K.
2. Peak and total VRAM footprint as a function of K.
3. Tenants concurrently served on one GPU, and per-tenant latency, at residency levels that permit co-location versus the maximum-fit level that does not.
4. Residency transition cost, separated by direction, since offloading and restoring are not symmetric.

Secondary metric, reported as a paired hypothesis test: deadline-miss rate under injected compute contention, comparing an adaptive residency policy against the fixed static policy. A paired Wilcoxon signed-rank test with matched-pairs rank-biserial effect size, six repetitions, alpha of 0.05.

Power caveat, stated in advance: a single model family and a small sample size mean only large effects are reliably detectable.

Null-result framing, fixed in advance: a finding that residency adaptation does not help in a given regime is reported as exactly that. Establishing where a mechanism fails bounds the claim and is a reportable outcome.

**Contention scenarios.** Two distinct forms of resource pressure, because they act on different resources:
1. Compute contention: a duty-cycled saturating GEMM workload in an independent process sharing the GPU through CUDA MPS, modeled on Kutukcu et al. (`literature-survey.md` source 6). Because fine-grained GPU occupancy cannot be measured in the target environment (see constraints below), contention intensity is defined as the duty cycle of that workload rather than read from a utilization counter.
2. Memory contention: an independent process holding GPU memory, reducing the budget available to the model.

## 2. Proposed Technical Approach

**Foundation.** The project forks `aveeslab/oom-free-alpamayo`. The swapping mechanism itself, Sequential and Pipelined Demand Layering, is correct existing infrastructure and is not modified. The residency decision is the surface under study.

1. Characterization: measure latency and VRAM footprint across the residency range on real inference, establishing the tradeoff curve the existing system leaves unexplored.
2. Co-location: determine how many model instances can share one GPU at residency levels below maximum-fit, and at what per-tenant latency cost, against the maximum-fit configuration that permits only one.
3. Memory-aware residency: a policy that reads available GPU memory and selects a residency level that fits, compared against the fixed offline choice under pre-existing memory pressure.
4. Boundary: an adaptive policy evaluated against the static baseline under compute contention, testing whether residency adaptation can mitigate compute interference at all.
5. Transition cost: measure the cost of changing residency at runtime, separated by direction, and evaluate whether the existing interleaved placement rule is appropriate when residency is no longer chosen once.

**Data sources.** The model, `nvidia/Alpamayo-R1-10B`, is open and ungated, licensed under OpenMDW-1.1. The codebase is MIT licensed. Representative driving-scene input frames come from NVIDIA's `PhysicalAI-Autonomous-Vehicles` dataset, which is gated by an automatic license agreement; six distinct clips are used, one per repetition, so that paired comparisons hold input constant within a repetition while varying scene content across repetitions.

**Technical modifications.** A memory-sensing residency selection policy replacing the fixed offline choice; a characterization harness measuring latency and footprint across residency levels; compute and memory contention generators; a multi-tenant co-location harness; and an evaluation pipeline.

## 3. Target Environment Constraints

The following were confirmed by direct test on the rented GPU and constrain what can be measured. They are stated here because they shape the metric definitions above.

- GPU graphics clock locking is denied by the hypervisor even with root and passwordless sudo. Clock frequency is therefore logged during runs so that throttling can be detected and disclosed rather than silently absorbed.
- The DCGM Profiling module, which provides fine-grained SM-occupancy counters, fails to load. Hardware performance counter access is commonly restricted in multi-tenant GPU environments.
- `nvidia-smi` reports `utilization.gpu` as 100 percent at every duty cycle from 20 to 100 percent, so it detects whether any kernel is active rather than fractional occupancy, and cannot be used to calibrate contention intensity.
- `torch.cuda.mem_get_info` is cross-process aware and is used as the memory-pressure sensing signal. It requires no elevated privilege, so a deployed system would have the same signal available.

## 4. Device Available and Maintainer

Device available: NVIDIA RTX 5050 (desktop) for code development; model loading, inference, and contention experiments run on a rented RTX 3090 (24GB VRAM), since Alpamayo-R1-10B's 22GB footprint exceeds the local card's memory. Maintainer: Uday Arora; Claude Code access is requested to support implementation throughout the semester.
