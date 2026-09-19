# AI Novelty and Feasibility Audit: GPU Residency Scheduling for Oversized VLA Models

This audit follows two adversarial passes, novelty and feasibility, searching actively for reasons the idea fails or already exists rather than reasons it works. It was revised twice: first after an initial experimental design was found untestable, and again after feedback on the submitted proposal directed a scope reduction, a simpler policy over a few pre-profiled configurations, and explicit evaluation of what changing residency costs.

Both revisions pointed the same way. The feedback warned that if layer migration latency dominates inference, a theoretically better placement may increase deadline misses. Measurement confirmed that risk is real and then located its largest single cause, which is described under Engineering lift below.

## AI Critique Summary

Is this genuinely novel? Yes, narrowly, and the honest scope is narrower than first claimed. The mechanism, CPU-GPU layer swapping to fit an oversized model, is not novel: a peer-reviewed system, `oom-free-alpamayo` (IEEE RTCSA 2026), already does it well, and a related system, Nova, applies the same core technique to desktop model serving. What is not found anywhere, after reading five related systems in full text, is treating the layer-residency count as a characterized runtime tradeoff rather than a fixed offline choice. The existing system picks one operating point by capacity arithmetic on an idle GPU and never revisits it; the curve that decision sits on has not been published.

An initial version of this project claimed a deadline-aware and contention-aware adaptive residency policy would reduce deadline-miss rate under compute contention. Direct measurement refuted the premise. Latency decreases monotonically with resident layer count, and the maximum-fit configuration is already the fastest, so the policy's only available action makes latency worse. Compute contention was further measured to slow all residency levels proportionally, leaving residency with no lever against it. That claim was withdrawn rather than defended, and the negative result is now reported as a boundary on where the mechanism applies.

The surviving claim is characterization: what the residency knob costs in latency, what it buys in memory footprint, which operating points the fixed offline choice forecloses, and which forms of resource pressure residency can answer. This is a smaller contribution than originally proposed and is stated as such.

## Novelty Audit: Survives with Named Limitations

**Why it survives.** Five related systems were read in full text (`literature-survey.md`, sources 1 through 5), including a check for whether `oom-free-alpamayo` itself has been extended or cited elsewhere. No source characterizes intra-model residency as a tunable latency and memory tradeoff, nor examines operating points other than maximum fit. Nova performs the swapping mechanism without examining the residency decision. RED and Action Chunk Scheduling address scheduling among already-resident models or across a robot fleet, not memory management within one oversized model. FASTER accelerates a different pipeline stage.

**Named limitation 1.** A close competitor published in June 2026 (`literature-survey.md`, source 5) addresses large driving VLA models against hardware limits through a runtime-adaptive compute-placement split, validated in real-vehicle experiments. The differentiation is the resource and the decision variable, not adaptivity: that work load-balances compute for a model that already fits, migrating a bounded set of boundary layers in response to utilization and queue signals. This project treats memory residency as the decision variable for a model that does not fit at all. The distinction sharpened under replanning, since the compute domain was measured here and found to be one residency cannot address.

**Named limitation 2.** CPU-GPU layer swapping is not novel. The claimed contribution is the characterization of the residency decision and its consequences, not the mechanism.

**Named limitation 3.** The 100ms real-time figure is sourced from Alpamayo's own paper, measured on a full-VRAM GPU, not the swapped system. Success criteria are defined relative to the system's own achievable range, not that figure.

**Named limitation 4, added after replanning.** The original adaptive-policy claim did not survive measurement. A design whose treatment arm cannot differ from its control arm cannot test its own hypothesis, and this one could not. The reported contribution was reduced accordingly rather than restated to fit the result.

## Feasibility Audit: Survives with Named Limitations

Based on verified resource facts and direct environment testing.

**Model and code access.** `nvidia/Alpamayo-R1-10B` is real and ungated, confirmed directly. `aveeslab/oom-free-alpamayo` is real, MIT licensed, and actively maintained, but depends on the `alpamayo_r1` package from `NVlabs/alpamayo`, which its own README does not name as a separate clone; this cost setup time to discover. The `nvidia/PhysicalAI-Autonomous-Vehicles` dataset used for driving-scene inputs is gated and additionally requires an access token with the public-gated-repository permission explicitly enabled, which is not granted by default.

**Compute access, named limitation.** Every experiment runs on a rented GPU, since the local development machine cannot hold the 22GB model. CUDA MPS was confirmed working, with two concurrent processes verified to genuinely share the GPU rather than serialize.

**Environment restrictions, confirmed by direct test.** Four capabilities assumed available at proposal time are not:
1. GPU graphics clock locking is denied by the hypervisor even with root and passwordless sudo, so timing runs cannot pin clocks. Clock frequency is logged instead so throttling is detectable and disclosable.
2. The DCGM Profiling module fails to load, so fine-grained SM-occupancy counters are unavailable. Hardware performance counter access is commonly restricted in multi-tenant GPU environments.
3. `nvidia-smi` reports `utilization.gpu` as 100 percent at every duty cycle from 20 to 100 percent, so it cannot calibrate contention intensity. Intensity is therefore defined as the duty cycle of the injected workload rather than measured.
4. cuTile requires driver CUDA 13.0 or newer; the host provides 12.7 and the RTX 3090 is not eligible for forward compatibility, so that toolchain is unavailable regardless of installation success.

These restrict how contention is specified and how timing variance is controlled. None prevents the characterization measurements, which depend only on wall-clock latency and `torch.cuda.mem_get_info`, both of which were verified working, the latter confirmed cross-process aware.

**Compute cost.** No model training, only a bounded evaluation sweep. Measured spend to date is under $3 on a rented RTX 3090, including a full experimental run that was discarded when its design was found invalid.

**Engineering lift, and the residency-change overhead.** The codebase being extended is small and its authors isolated the residency function.

Measuring what a residency change costs, as the feedback asked, produced the project's strongest result. Changing residency costs 0.3 to 4.7 seconds against 7 to 17 seconds of inference, so migration can indeed consume a large share of a deadline budget. The cost is also asymmetric: shedding layers is 3 to 6 times dearer than restoring them, because offloading pins host memory while restoring is a plain device copy.

The largest single cause is the placement rule itself. The upstream `interleaved_placement` is not nested across residency levels, so a change in K reshuffles layers that did not need to move. This is provable rather than incidental: a transition between residency sets costs the size of their symmetric difference, which is at least the change in count, with equality exactly when one set contains the other. The upstream rule violates that condition; its worst case moves 33 layers to change residency by 1. A nested rule built by recursive bisection meets the bound at every transition and produces bit-identical model outputs. Measured transition speedup ranges from 1.33x at coarse adjustments to 6.55x at the fine adjustments a switching policy actually makes on an RTX 3090, and the same shape holds on an RTX 4090 (1.28x to 8.44x) and an RTX 5070 Ti (1.65x to 13.68x). Those figures are not a hardware ranking, because the adaptation paths are generated from each card's maximum residency and so differ between runs. Holding the paths fixed does isolate hardware: with move counts identical at every granularity, an RTX 4090 gains 21.84x against an RTX 5070 Ti's 13.68x at a single-level adjustment, 7.63x against 4.48x at two levels, and the two converge to parity by eight levels and cross by twelve. The benefit of eliminating redundant moves is therefore largest where moves are most expensive and where the adjustment is finest, which is the regime a switching policy operates in.

The rule is not free, and the cost falls at low residency. Spread parity with upstream holds at K=2, 4, 8, 12 through 17, and 19 through 33, which includes all three pre-profiled operating points (16, 24 and 33). It fails at K=3, 5, 6, 7, 9, 10 and 18, where the nested rule spreads worse, and at K=11, where it spreads better. Parity of this kind has to be checked by sweeping the whole range rather than sampling it: the residency levels of most interest, 8, 16, 20, 24, 30 and 33, are ties without exception, so any subset drawn from them reports parity everywhere and conceals all eight levels where it does not hold. Steady-state latency parity is a property of the card, not of the residency level. The two rules match to a mean of 0.098 percent on an RTX 3090 and 0.054 percent on an RTX 4090 across levels 16 to 33. At the low levels 7, 10 and 14 an RTX 5070 Ti shows the nested rule slower by a mean of 1.028 percent, but an RTX 4090 run at those same levels, on the same six clips, matches to 0.093 percent. Low residency is therefore not the cause, and the penalty seen on the 5070 Ti is specific to that card. Spread does not account for it either, since the level with the worst nested spread of the three, K=10, is the one level where that card shows no penalty at all.

An ablation separates the two properties: a sequential nested order is equally move-optimal but distributes layers poorly, so nesting and spread are independent and the contribution is a rule achieving both rather than the observation that nesting helps.

This matters for the feedback's specific concern. Migration overhead can undermine adaptive residency, and a substantial part of that overhead was an artifact of a placement rule designed for a decision made once offline.

**Credibility of the foundation.** The codebase is the released artifact of a paper accepted at IEEE RTCSA 2026, an established venue in embedded and real-time systems.
