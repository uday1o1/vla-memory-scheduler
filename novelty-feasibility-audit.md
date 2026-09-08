# AI Novelty and Feasibility Audit: Deadline and Contention Aware GPU Memory Scheduling for VLA Models

This audit follows two adversarial passes, novelty and feasibility, searching actively for reasons the idea fails or already exists rather than reasons it works.

## AI Critique Summary

Is this genuinely novel? Yes, narrowly. The mechanism, CPU-GPU layer swapping to fit an oversized model, is not novel: a peer-reviewed system, `oom-free-alpamayo` (IEEE RTCSA 2026), already does it well, and a related system, Nova, applies the same core technique to desktop model serving. What is not found anywhere, after reading five related systems in full text and running a dedicated search for closer prior art, is making the layer-residency decision itself a live function of a deadline budget and sensed GPU contention, for a VLA model specifically. One close competitor, a static compute-placement approach published in June 2026, addresses the same underlying problem through a different mechanism and must be cited and differentiated rather than ignored. This project should not claim it achieves true real-time inference; even the best-resourced 2026 competitor does not reach that bound. Success is framed as relative improvement over the existing static baseline. The codebase being extended is small, from a single paper, real and peer-reviewed; its documented extension point should be verified early in implementation to confirm it is as cleanly isolated in practice as its documentation suggests.

## Novelty Audit: Survives with Named Limitations

**Why it survives.** Five related systems were read in full text (literature-survey.md, sources 1 through 5), and a dedicated search, including a check for whether `oom-free-alpamayo` itself has already been extended or cited elsewhere, found no source combining deadline awareness, live GPU-contention awareness, and intra-model memory-residency decisions for a VLA model. Nova performs the swapping mechanism without deadline or contention framing. RED and Action Chunk Scheduling address deadline awareness at a different layer, scheduling among already-resident models or across a robot fleet, not memory management for one oversized model. FASTER accelerates a different part of the inference pipeline entirely.

**Named limitation 1.** A close competitor published in June 2026 (literature-survey.md, source 5) addresses the same underlying problem, large driving VLA models against real hardware limits, through a static CPU/GPU compute-placement split validated once on real driving VLA models. This project's contribution is stated precisely relative to it: a live-adaptive memory-residency policy reacting to a sensed deadline and contention signal at runtime, not a static, offline-validated placement decision. This source was read at abstract level only; a full text read is planned before implementation to confirm the distinction holds.

**Named limitation 2.** CPU-GPU layer swapping is not itself novel. The claimed contribution is the policy governing residency decisions under deadline and contention pressure, and this distinction should be stated explicitly in any presentation of this work.

**Named limitation 3.** The 100ms real-time figure is sourced from Alpamayo's own paper, measured on a full-VRAM GPU, not the swapped system and not a realistic rented GPU. Success criteria are defined relative to the static baseline's own achievable range rather than to this figure directly, which is the correct framing to preserve.

## Feasibility Audit: Survives with Named Limitations

Based only on verified resource facts, not duration estimates.

**Model and code access.** `nvidia/Alpamayo-R1-10B` is real, ungated, and permissively licensed; its weight size of 22GB was confirmed directly, which supports the premise that it does not fit on a 12 to 16GB GPU. `aveeslab/oom-free-alpamayo` is real, substantive, MIT licensed, and actively maintained. Its upstream dependencies are real, public, and Apache-2.0 licensed.

**Compute access.** Testing the contention-aware half of this project requires running two processes on one GPU simultaneously, which requires CUDA MPS administrative access. MPS support inside a container is provider-dependent; a provider offering root or full-OS pod access is required, confirmed through direct testing before committing to any provider.

**Compute cost.** This project involves no model training, only a bounded evaluation sweep of baseline against adaptive policy across two contention scenarios and at least eight repetitions each. Estimated at $40 to $150 on a rented A100-class GPU.

**Data access.** The model and code require no special access. Representative driving-scene inputs are available either through NVIDIA's automatically-gated driving dataset, needed only as a small sample, or through a community integration with the CARLA simulator, which avoids the licensed dataset question entirely.

**Engineering lift.** The codebase being extended is small, and its authors have already isolated the exact function this project modifies, rather than requiring that boundary to be identified from scratch.

**Credibility of the foundation.** The codebase is not an unreviewed hobby project; it is the released artifact of a paper accepted at IEEE RTCSA 2026, an established venue in embedded and real-time systems.
