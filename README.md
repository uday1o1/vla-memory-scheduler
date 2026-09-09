# Deadline and Contention Aware GPU Memory Scheduling for Vision-Language-Action Models

**Course:** CMPE 249, Intelligent Autonomous Systems, Fall 2026, Prof. Kaikai Liu
**Team:** Uday Arora
**Track:** Deployment Track

## Abstract

Vision-Language-Action (VLA) models for autonomous driving, such as NVIDIA's open 10-billion-parameter Alpamayo, are too large to fit in the memory of commodity and automotive-grade GPUs. A peer-reviewed system, `oom-free-alpamayo` (IEEE RTCSA 2026), fixes the fit problem by streaming model layers between CPU and GPU on demand, but the specific set of layers it keeps GPU-resident is decided once, offline, purely to maximize throughput, with no notion of a control-loop deadline and no awareness of a second workload sharing the GPU. The result is an inference time far outside real-time bounds for autonomous driving, a gap this project does not aim to close outright, since no existing system in this space reaches true real-time. Instead, this project makes the residency decision itself runtime-adaptive, responding to a live deadline budget and to sensed GPU contention, and asks a narrower, answerable question: does that adaptivity measurably reduce the deadline-miss rate and degrade more gracefully under contention than the existing static policy. Full numbers and sources: `proposal.md` and `literature-survey.md`.

## Repository Contents

- `literature-survey.md`: SOTA survey (Deliverable B)
- `proposal.md`: problem formulation, technical approach, device and maintainer (Deliverable C)
- `novelty-feasibility-audit.md`: AI novelty and feasibility audit (Deliverable D)

## Foundation

This project extends `aveeslab/oom-free-alpamayo`, the open-source release accompanying its IEEE RTCSA 2026 paper, which itself depends on NVIDIA's open Alpamayo model and inference source. Full citations and links are in `literature-survey.md`.
