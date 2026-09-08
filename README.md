# Deadline and Contention Aware GPU Memory Scheduling for Vision-Language-Action Models

**Course:** CMPE 249, Intelligent Autonomous Systems, Fall 2026, Prof. Kaikai Liu
**Team:** Uday Arora
**Track:** Deployment Track. The track's own definition, optimizing a model for latency, memory, quantization, batching, or edge compute, is a direct match for this project's content.

## Abstract

Vision-Language-Action (VLA) models for autonomous driving, such as NVIDIA's open 10-billion-parameter Alpamayo, exceed the memory capacity of commodity and automotive-grade GPUs (22GB of weights against a typical 12 to 16GB budget). CPU-GPU layer swapping, as implemented in the peer-reviewed system `oom-free-alpamayo` (IEEE RTCSA 2026), resolves this capacity mismatch but at a severe latency cost: reported inference times of 4.09 to 15.46 seconds, against a stated real-time requirement of approximately 100 milliseconds for autonomous driving. The cause is that the system's layer-residency policy is computed once, offline, to maximize throughput, without regard for an actual control-loop deadline or for contention from other workloads sharing the GPU. This project proposes a runtime-adaptive residency policy that responds to a live deadline budget and to sensed GPU contention, evaluated against the existing static policy under synthetic contention scenarios. Success is defined as a statistically significant reduction in deadline-miss rate and improved degradation behavior under contention, not as achieving true real-time inference, since no existing system in this space reaches that bound.

## Repository Contents

- `literature-survey.md`: SOTA survey (Deliverable B)
- `proposal.md`: problem formulation, technical approach, device and maintainer (Deliverable C)
- `novelty-feasibility-audit.md`: AI novelty and feasibility audit (Deliverable D)

## Foundation

This project extends `aveeslab/oom-free-alpamayo`, the open-source release accompanying its IEEE RTCSA 2026 paper, which itself depends on NVIDIA's open Alpamayo model and inference source. Full citations and links are in `literature-survey.md`.
