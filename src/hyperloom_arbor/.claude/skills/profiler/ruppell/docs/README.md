# docs

| File | What |
| --- | --- |
| [status.md](status.md) | What has been verified, per platform and workload, with the record behind each claim |
| [limitations.md](limitations.md) | What the traces and the census cannot be trusted for yet, and why |
| [validation.md](validation.md) | Every validation run so far, numbers only: against rocprofv3 and Nsight Systems, against torch.profiler, attribution and steps, whether tracing changes the computation, cost, copies, NVTX and ROCTX ranges, collectives, and what validation found wrong |
| [nsys-comparison.md](nsys-comparison.md) | ruppell against Nsight Systems, feature by feature: what ruppell lacks (copies, collective metadata, counters, the CPU side, NVTX, a viewer), where it goes further, and what to do first |
| [birdeye.md](birdeye.md) | birdeye, the built-in micro TraceLens: its sources, its tables and how they map to TraceLens's |
| [background.md](background.md) | Why PyTorch traces mislead on graph-replayed serving: eight defects, measured, with root causes |
| [design.md](design.md) | The three layers, what "owning the runtime" means, the cost of owning it, open questions (design of record) |
| [plan-v0.md](plan-v0.md) | The plan that builds and validates v0 (plan of record; formerly the runner's Plan 23) |
| [roadmap.md](roadmap.md) | Milestones M0–M4; M1 is a working Chrome trace generator, broken into steps with exit criteria |
| [roadmap-v2.md](roadmap-v2.md) | Milestones M6–M14: the gaps to Nsight Systems (copies and fills, NVTX/ROCTX, collectives, counters, CUDA breadth, a viewer and export, external control and many nodes), ruppell's own cost, and components profiled in depth by default |
| [gray-parrot/README.md](gray-parrot/README.md) | gray-parrot, the hardware-counter module (M9, off by default): per kernel on counter steps, per step, and device activity over the run; its design and milestones GP0–GP5 |
| [gray-parrot/survey.md](gray-parrot/survey.md) | What Nsight Systems, Nsight Compute, CUPTI, rocprofv3 and the ROCm profiler SDK offer for counters, with gfx950's counter list and a counter smoke on MI350X |
| [gray-parrot/records.md](gray-parrot/records.md) | gray-parrot's records, the contract ROCm and CUDA both write: `counters.bin`, `device_counters.bin`, `counters.tsv`, `meta.json`'s `counters`, the counter step's field |
| [feasibility.md](feasibility.md) | What the ROCm profiler SDK provides and how ruppell uses it (M1.1 answers, with the jobs that showed them) |
| [milestones/m1.md](milestones/m1.md) | M1 record: every exit criterion with its evidence, the build it came from, cost, limits |
| [milestones/accuracy-study.md](milestones/accuracy-study.md) | The accuracy study (2026-10-20): against rocprofv3 in the same process (identical, every dispatch), Nsight Systems on B200, and torch.profiler on MI350; what it changed |
| [milestones/dlrmv4.md](milestones/dlrmv4.md) | A second workload: DLRMv4 training on MI350X. The gates with their evidence, what training needed (backward attribution, forked workers, start-up waits), the witness on a torch-bundled runtime, program identity under autotuning |
| [analysis/dlrmv4-training-profile.md](analysis/dlrmv4-training-profile.md) | What the DLRMv4 traces show: the step decomposed, the FP8 forward's and backward's mechanisms kernel by kernel, what is left in bf16, the metrics step, against `torch.profiler` |
| [plan-m2.md](plan-m2.md) | M2 plan: the vLLM adapter, the serving schema, per-process runs, the census, on DeepSeek V4.1 Flash (draft) |
| [plan-cuda.md](plan-cuda.md) | CUDA plan: the runtime layer on B200 through CUPTI, the node-id join, the witness question, risks and effort (draft) |
| [related-tools.md](related-tools.md) | TraceLens's Analysis agent and Hyperloom's phase-coverage fix: what they do on vLLM serving, their limits, and why ruppell does it differently |
| [tracelens.md](tracelens.md) | Reading ruppell traces with AMD TraceLens (`tracelens`, `export --format torch`), its limits, and what the MoE trace shows |
| [cpu-ops.md](cpu-ops.md) | The CPU op tree (`RUPPELL_CPU_OPS`): aten ops and module scopes on host threads, every kernel to its op; cost; what it shows on the MoE |
| [upstream.md](upstream.md) | What of ruppell can go to PyTorch and the ROCm profiler SDK, and what cannot: the SDK's new graph ids, Kineto #1567, PyTorch's capture-time annotations, and where the rest belongs |
