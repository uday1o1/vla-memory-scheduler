# Project Chat Starter Prompt

Paste this into a fresh chat to begin (or resume) implementation work on this project with full context, without carrying over any unrelated prior-project history.

---

I'm working on my CMPE 249 (Intelligent Autonomous Systems) semester project at SJSU, taught by Prof. Kaikai Liu. Treat this as the start of this project's working history in this chat.

**Project:** Deadline and Contention Aware GPU Memory Scheduling for Vision-Language-Action Models
**Repository:** https://github.com/uday1o1/vla-memory-scheduler (cloned locally at `/Users/uday1o1/Documents/SJSU/Sem3/IAS/vla-memory-scheduler`)
**Track:** Deployment Track — a direct, literal match to CMPE 249's own "optimize a model for latency, memory... edge compute" wording.

**Abstract:** NVIDIA's Alpamayo (a real, open 10B-parameter Vision-Language-Action model for autonomous driving, 22GB) doesn't fit on 12-16GB GPUs. A real, peer-reviewed system (`oom-free-alpamayo`, IEEE RTCSA 2026) fixes this via CPU-GPU layer swapping, but its layer-residency policy is static, offline, and throughput-only — resulting in 4-15 second inference against a field-standard ~100ms real-time requirement (sourced from Alpamayo's own paper). This project makes that residency policy deadline-aware and contention-aware (reacting to a real control-loop budget and to a second workload sharing the GPU) instead of static — extending the system's own explicitly-named limitation. Success is framed honestly as relative improvement over the static baseline, not "achieving real-time," since even the best 2026 competitor doesn't get there either.

**This project replaced an earlier CMPE 249 topic** (concurrent GPU scheduling for radar/ISAC + perception workloads) after the student determined the domain — not the technical style — was the reason to move on. The old project is fully archived, not deleted, at `/Users/uday1o1/Documents/SJSU/Sem3/IAS/research/archive-old-gpu-scheduling-project/`. Do not reference or resemble it in this project's framing.

The repository already contains a completed, audited proposal:
- `README.md` — title, abstract, track
- `literature-survey.md` — 8 sources, verification level stated per source (full-text vs. abstract-level)
- `proposal.md` — problem formulation, technical approach, device/maintainer
- `novelty-feasibility-audit.md` — two-stage adversarial audit, including named limitations that are still-open risks

Please read all four files first so you have full technical context before we do anything else.

**Standards to hold for all future work in this project, not just the proposal phase:**
- Never anchor a technical claim on a single source; require multiple independent, real, primary sources.
- Read primary sources directly (papers, docs, code) rather than relying on memory or abstracts; quote or closely paraphrase what you actually find, with real links. If something was only checked at abstract/snippet level, say so explicitly rather than presenting it as full-text-verified.
- If you ever cite a bug report, GitHub issue, or forum thread as evidence, read the full thread and state its real resolution status explicitly.
- Keep the mechanism-vs-contribution distinction sharp: CPU-GPU layer swapping is not novel by itself (OOM-Free Alpamayo, Nova both already do it) — the contribution is specifically the deadline- and contention-aware *policy* governing residency decisions, not the swapping mechanism.
- Do not claim this project achieves true real-time (~100ms) inference for Alpamayo — even the best-resourced 2026 competitor found in research doesn't reach that. Success criteria are relative improvement over the existing static baseline and graceful-degradation-under-contention, fixed in proposal.md §1 before data collection.
- A newly-found close competitor (literature-survey.md #5, "Efficient Block-Layer Parallel Inference for VLA on Hybrid Architectures") was only checked at abstract level — read its full text early in implementation to confirm it truly never adapts its compute-placement decision at runtime, since that's the exact distinction this project's novelty claim rests on.
- Base feasibility/scope decisions only on verified resource facts (real compute cost, real data/tooling access) — never on duration/time estimates.
- MPS administrative access for the contention-injection experiments is provider-dependent (a finding carried over from this student's prior GPU-scheduling project) — RunPod is the confirmed-reliable choice; run the pre-registration smoke test before committing to any provider.
- The model, code, and upstream dependencies (`nvidia/Alpamayo-R1-10B`, `aveeslab/oom-free-alpamayo`, `NVlabs/alpamayo1.5`, `NVlabs/alpamayo`) are all real, public, and already cloned at `/Users/uday1o1/Documents/SJSU/Sem3/cloned-repos/`. The driving dataset (`nvidia/PhysicalAI-Autonomous-Vehicles`) is auto-gated but its license "Purpose" clause is a plausible-not-certain fit for this kind of systems study — only pull a small sample, and consider the `alpamayo-carla-bridge` community project (also cloned locally, verified real/substantive) as an alternative that sidesteps the dataset-license question entirely.

Start by reading the repo's four files, then propose a concrete implementation plan for the next milestone (understanding and instrumenting `oom-free-alpamayo`'s existing residency-decision function, then building the deadline/contention-sensing layer on top of it), flagging anything in the existing proposal you think needs revisiting before we start building.
