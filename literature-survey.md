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

Read in full text. Partitions a driving VLA model's backbone so the language model suffix runs on CPU through an asynchronous cross-frame pipeline, tested on two real driving VLA models under coexistence with the rest of an onboard compute stack, including real-vehicle experiments running concurrently with Autoware.Universe. Reports 21.7 to 30.9 percent latency reduction and comparable peak memory reduction. This paper does adapt at runtime: it defines a state vector of GPU utilization, CPU utilization, and pipeline queue backlog, and uses it to migrate a small number of layers near a fixed partition boundary online. The differentiation is not "static versus adaptive" but what is being solved and what drives the adaptation. This paper solves compute placement, load-balancing an already-fits model across CPU and GPU for speed, reacting to raw utilization and queue signals, with only a bounded set of boundary layers eligible to move. This project solves memory residency, fitting a model that does not fit in GPU memory at all, reacting to an explicit deadline budget and a GPU-contention signal, with the full residency decision recomputed rather than a small fixed zone. Both are genuine runtime-adaptive systems; they adapt to different signals for different underlying problems.

### 6. Kutukcu, Baidya, Raghunathan, Dey. "Contention Grading and Adaptive Model Selection for Machine Vision in Embedded Systems." ACM Transactions on Embedded Computing Systems, 2022. DOI: [10.1145/3520134](https://doi.org/10.1145/3520134)

Read at abstract level. This paper's contention-generation approach, profiling the system under a dummy GPU workload to produce a controllable, steady contention level, is the technique this project's synthetic contention generator is modeled on.

### 7. Grover et al. "Embodied Foundation Models at the Edge: A Survey of Deployment Constraints and Mitigation Strategies." [arXiv:2603.16952](https://arxiv.org/abs/2603.16952)

Read at abstract level only. Frames VLA deployment at the edge as a systems problem and states that autoregressive VLA policies are memory-bandwidth bound, while diffusion-based controllers are compute-latency bound. Cited only as corroborating context for this being a recognized problem class.

### 8. Oneiros. ACM SoCC 2025, pp. 88-101. DOI: [10.1145/3772052.3772215](https://doi.org/10.1145/3772052.3772215). [arXiv:2507.11507](https://arxiv.org/abs/2507.11507). Code: [github.com/UT-SysML/Oneiros](https://github.com/UT-SysML/Oneiros)

Read in full text, with the source checked. The closest prior work to changing residency at runtime. Inside vLLM, when the KV cache is exhausted, it remaps the memory of some weight layers to KV cache and streams those layers from a pinned host copy, revisiting the decision at every scheduler step and reverting off-peak. Layers are chosen at uniform intervals (`step_size = total_layers // (cpu_layers + 1)`), which is the same evenly spaced family as the upstream rule here and is not nested across counts. Because it keeps a complete host copy, evicting a layer moves no data. Its optimality argument covers steady-state transfer only: it does not report how long a resize takes, and it has no deadline metric. Evaluated on OPT and Llama models from 6.7 to 30 billion parameters on a GH200. A controller that re-chooses the resident count from free memory with evenly spaced placement and a host copy is the strong baseline this project compares against.

### 9. ElastiLM, "Elastic On-Device LLM Service." ACM MobiCom 2025. DOI: [10.1145/3680207.3765259](https://doi.org/10.1145/3680207.3765259). [arXiv:2409.09071](https://arxiv.org/abs/2409.09071)

Read in full text. Reorders weights offline by importance so that each smaller sub-model is a contiguous prefix of the larger, which makes switching between them free: only a pointer moves. This is the same principle as nested residency, that a nested family of configurations makes changing between them cheap, applied to pruned sub-models whose outputs differ from one another, rather than to the placement of identical weights.

### 10. Mattson, Gecsei, Slutz, Traiger. "Evaluation Techniques for Storage Hierarchies." IBM Systems Journal 9(2):78-117, 1970. DOI: [10.1147/sj.92.0078](https://doi.org/10.1147/sj.92.0078)

Bibliographic record verified; the 1970 full text was not obtained, so the property is stated as given by Gu and Ding ([ISMM 2011](https://www.cs.rochester.edu/~cding/Documents/Publications/ismm11a.pdf)). Defines stack algorithms by the inclusion property: at every point, the contents of a memory of capacity C are a subset of the contents at capacity C+1. A single priority order that does not depend on capacity gives inclusion. Nested residency is this property applied to resident layers. The move-count bound this project states follows from it by elementary set arithmetic and is not claimed as new.

### 11. Host-authoritative weight offloading: FlexGen, DeepSpeed ZeRO-Inference, Hugging Face Accelerate, llama.cpp, vLLM, PipeSwitch

Checked against source code: FlexGen [arXiv:2303.06865](https://arxiv.org/abs/2303.06865) (`flex_opt.py`), DeepSpeed (`partition_parameters.py`), Accelerate (`AlignDevicesHook` in `hooks.py`), llama.cpp (`ggml-backend.cpp`) and vLLM `--cpu-offload-gb` (`offloader/uva.py`). PipeSwitch (OSDI 2020) was read in full text. In all of them the host holds the authoritative copy of the weights and the device copy is disposable, so freeing device weights needs no transfer back. The upstream system instead moves a resident layer's only copy to the device and, to demote it, copies it back and pins it. Keeping a pinned host copy in that system is therefore a correction to standard practice, not a contribution. Accelerate's `device_map` and llama.cpp's `-ngl` happen to place a prefix or suffix of layers, which is nested, but both fix placement at load time.

### 12. Related runtime reconfiguration. Select-N [arXiv:2502.08182](https://arxiv.org/abs/2502.08182); SpotServe, ASPLOS 2024, [arXiv:2311.15566](https://arxiv.org/abs/2311.15566); LiteReconfig, EuroSys 2022, DOI: [10.1145/3492321.3519577](https://doi.org/10.1145/3492321.3519577)

Read at abstract and method level. Select-N offloads every i-th layer and adjusts i each iteration for PCIe contention, under a cap set by memory; its families are nested only along multiples of i. SpotServe minimizes migration between parallel configurations by matching, rather than by designing the configurations to nest. LiteReconfig charges a switching cost that depends on the pair of configurations but does not shape the configuration space to reduce it. None measures the cost of a resize against a deadline.

---

**Summary of the gap.** CPU-GPU layer swapping for oversized models is established (sources 0, 1 and 11). Changing residency at runtime in response to memory pressure is also established: Oneiros does it inside an LLM server (source 8). Nesting configurations so that switching between them is cheap is an old principle (source 10) with a recent instance for pruned sub-models (source 9), and keeping a host copy so that eviction is free is standard practice (source 11). What is not found in any of them: that the evenly spaced placement used by the upstream system, by Oneiros and by Select-N is far from nested when residency changes at runtime, needing 33 layer moves for a change of one; what that costs when measured on a real VLA model with identical outputs; and whether the cost of changing residency makes a policy that tracks free memory miss deadlines that a static choice would meet. No surveyed source evaluates residency changes against a deadline.
