# Deadline and Contention Aware GPU Memory Scheduling for Vision-Language-Action Models

**Course:** CMPE 249 — Intelligent Autonomous Systems, Fall 2026, Prof. Kaikai Liu
**Team:** Uday Arora
**Track:** Deployment Track — per CMPE 249 Lecture 1's official track menu ("Optimize a model for latency, memory, quantization, batching, edge compute, or multi-sensor synchronization"). This project is directly about latency and memory optimization for a real model under real hardware constraints — a more literal fit than this student's prior System-track self-classification on a different project.

## Abstract

NVIDIA's Alpamayo is a real, open, 10-billion-parameter Vision-Language-Action (VLA) model for autonomous driving. Its weights are 22GB (independently verified) — too large for the 12-16GB VRAM budgets common on automotive-grade and consumer GPUs. A real, peer-reviewed system (`oom-free-alpamayo`, IEEE RTCSA 2026) fixes the fit problem by streaming model layers between CPU and GPU memory on demand, keeping only the most valuable layers permanently GPU-resident. It works — a 22GB model now runs on a 12GB card — but its own reported inference time is 4.09-15.46 seconds, while Alpamayo's own paper states autonomous driving needs ~100ms end-to-end latency to be real-time: a 40-150x gap. The reason is that the system's layer-residency decision is made once, offline, purely to maximize throughput — it has no notion of an actual control-loop deadline, and no awareness of whether another workload is simultaneously competing for the same GPU. This project replaces that static, throughput-only policy with one that adapts at runtime to (a) a control-loop deadline budget and (b) live-sensed contention from a second, independently co-located GPU workload — extending the system's own explicitly-stated limitation, not the bare idea of layer swapping (which is not novel by itself). Full audit trail and every citation's verification status: `literature-survey.md` and `novelty-feasibility-audit.md`.

## Repository Contents

- `literature-survey.md` — SOTA survey with real links, verification level stated per source
- `proposal.md` — problem formulation, technical approach, device/maintainer
- `novelty-feasibility-audit.md` — two-stage adversarial audit (novelty + feasibility), including named limitations
- `PROJECT_CHAT_START.md` — onboarding prompt for continuing implementation in a fresh chat

## Reference Code (cloned, not redistributed — see `.gitignore`)

This project builds on real, external open-source code, cloned locally for development at `/Users/uday1o1/Documents/SJSU/Sem3/cloned-repos/`: `oom-free-alpamayo` (the codebase being extended), `alpamayo1.5` and `alpamayo` (NVIDIA's upstream model source, both required dependencies), `alpamayo-recipes` (NVIDIA's official fine-tuning/deployment recipes), and `alpamayo-carla-bridge` (a community CARLA-simulator integration, a candidate route for closed-loop evaluation without needing NVIDIA's licensed driving dataset).
