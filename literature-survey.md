# Literature and SOTA Survey: Deadline and Contention Aware GPU Memory Scheduling for VLA Models

Sources below were retrieved and read directly. Verification level (full text or abstract only) is stated per source.

### 0. Roh, Kim, Kim (AVEES Lab, Kookmin University). "OOM-Free Alpamayo via CPU-GPU Memory Swapping for Vision-Language-Action Models." IEEE RTCSA 2026. [arXiv:2605.11678](https://arxiv.org/abs/2605.11678). Code: [github.com/aveeslab/oom-free-alpamayo](https://github.com/aveeslab/oom-free-alpamayo)

Read in full text. Primary technical foundation of this project. Fits NVIDIA's 22GB Alpamayo-R1-10B and Alpamayo-1.5-10B models onto 12 to 16GB GPUs using three techniques: Sequential Demand Layering (streams layer weights on demand, reducing peak VRAM from model level to layer level), Pipelined Demand Layering (a double buffer and dedicated prefetch CUDA stream overlap the host to device transfer with compute), and a GPU-Resident Layer Decision Policy (a one-time offline profiling run selects which layers stay permanently GPU-resident to maximize throughput within a VRAM budget). Reported results: 4.09 seconds on an RTX 5070 Ti (16GB) and 15.46 seconds on an RTX 3080 Ti (12GB); both configurations run out of memory without this system. The residency policy is static, computed once per machine, and optimizes purely for throughput, with no deadline parameter and no live sensing of contention from another process. This is the gap the present project addresses. The code is real, substantive, and MIT licensed, released alongside the peer-reviewed paper.

### 0b. NVIDIA. "Alpamayo: A Foundation Model for Physical AI Applied to Autonomous Driving." [arXiv:2511.00088](https://arxiv.org/abs/2511.00088)

Read in full text. Source of this project's real-time deadline figure. States directly that real-time performance requires 99ms latency and that "the decoding process must be fast enough to support real-time inference... within the real-time requirements for autonomous driving (typically 100ms)." This figure was measured on a full-VRAM NVIDIA RTX 6000 Pro Blackwell GPU with no memory swapping. The swapped system above is 40 to 150 times slower than this figure, which is why the present project's success criteria are framed as relative improvement rather than as meeting a 100ms target.

### 1. "Nova." [arXiv:2509.21301](https://arxiv.org/abs/2509.21301)

Read in full text. The closest mechanism level prior work. Implements layer-wise CPU-GPU weight offloading for a vision transformer encoder using asynchronous CUDA streams, pinned memory, and CUDA events, the same mechanism family as OOM-Free Alpamayo. Scoped to GUI and agentic vision language model serving on desktop GPUs. Its real-time framing addresses its own bursty request stream, not resistance to an independent second workload. No mention of a robot, a VLA model, a control loop, or an external contention source appears in the paper. This is necessary prior art for the swapping mechanism; it does not address deadline or contention adaptivity.

### 2. "RED: Adaptive Real-Time DAG Scheduling for Robotic Inference." [arXiv:2605.24044](https://arxiv.org/abs/2605.24044)

Read in full text. Deadline-aware scheduling for multiple already GPU-resident DNN tasks on Jetson-class embedded GPUs, using weight sharing across tasks. Treats out-of-memory as an exception to avoid, not something to manage through layer streaming. This operates at a different layer: scheduling among tasks that already fit in memory, not managing memory for a single model that does not fit. No overlap with the present project.

### 3. "FASTER." [arXiv:2603.19199](https://arxiv.org/abs/2603.19199)

Read in full text. Accelerates the flow matching action generation steps inside a VLA model, a different stage of the pipeline than memory placement. Contains one passing mention of memory contention in its limitations. No overlap.

### 4. "Action Chunk Scheduling for Batched Robot Policy Serving." [arXiv:2608.00337](https://arxiv.org/abs/2608.00337)

Read in full text. Schedules which robot in a fleet is served from a shared remote GPU, an inter-robot scheduling problem rather than intra-model memory management. No overlap.

### 5. "Efficient Block-Layer Parallel Inference for Vision-Language-Action on Hybrid Architectures." [arXiv:2608.14586](https://arxiv.org/abs/2608.14586)

Read in full text directly after an earlier abstract-level pass mischaracterized it as static; corrected here. Partitions a driving VLA model's backbone so the language model suffix runs on CPU through an asynchronous cross-frame pipeline, tested on two real driving VLA models under coexistence with the rest of an onboard compute stack, including real-vehicle experiments running concurrently with Autoware.Universe. Reports 21.7 to 30.9 percent latency reduction and comparable peak memory reduction. This paper does adapt at runtime: it defines a state vector of GPU utilization, CPU utilization, and pipeline queue backlog, and uses it to migrate a small number of layers near a fixed partition boundary online. The differentiation is not "static versus adaptive" but what is being solved and what drives the adaptation. This paper solves compute placement, load-balancing an already-fits model across CPU and GPU for speed, reacting to raw utilization and queue signals, with only a bounded set of boundary layers eligible to move. This project solves memory residency, fitting a model that does not fit in GPU memory at all, reacting to an explicit deadline budget and a GPU-contention signal, with the full residency decision recomputed rather than a small fixed zone. Both are genuine runtime-adaptive systems; they adapt to different signals for different underlying problems.

### 6. Kutukcu, Baidya, Raghunathan, Dey. "Contention Grading and Adaptive Model Selection for Machine Vision in Embedded Systems." ACM Transactions on Embedded Computing Systems, 2022. DOI: [10.1145/3520134](https://doi.org/10.1145/3520134)

Read at abstract level. This paper's contention-generation approach, profiling the system under a dummy GPU workload to produce a controllable, steady contention level, is the general technique this project's synthetic contention injection is modeled on. The paper's own exact terminology for this mechanism was not independently confirmed against its full text, which is paywalled; this project's implementation is described in `proposal.md` as a synthetic contention generator, not attributed to a specific named term from the source.

### 7. Grover et al. "Embodied Foundation Models at the Edge: A Survey of Deployment Constraints and Mitigation Strategies." [arXiv:2603.16952](https://arxiv.org/abs/2603.16952)

Read at abstract level only. Frames VLA deployment at the edge as a systems problem and states that autoregressive VLA policies are memory-bandwidth bound, while diffusion-based controllers are compute-latency bound. Cited only as corroborating context for this being a recognized problem class.

---

**Summary of the gap.** CPU-GPU layer swapping for oversized models is established (sources 0 and 1). Deadline-aware scheduling exists at the inter-task and inter-robot level (sources 2 and 4), not at the level of memory-residency decisions inside a single model. The most recent related work (source 5) addresses the same domain problem through a static compute-placement mechanism rather than a live-adaptive one. No surveyed source combines deadline awareness, sensed GPU contention, and intra-model layer-residency decisions for a VLA model.
