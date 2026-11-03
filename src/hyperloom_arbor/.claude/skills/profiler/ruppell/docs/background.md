# Background: why PyTorch traces mislead on graph-replayed serving

> Copied on 2026-09-29 from `jiaweic-runner`, `plans/arbor/M03_GRAPH_AWARE_TRACER.md`
> sections 1–3 (the root-cause analysis). Plan numbers and paths refer to that
> repository. Measurements are from DeepSeek V4.1 Flash, AgentX, seed 42, on
> AMD Instinct MI355X.

## 1. The answer in one paragraph

The defects are mostly not PyTorch bugs. The PyTorch profiler (Kineto) is an
op-level, CPU-anchored tracer used in short windows. This workload's cost sits
where such a tracer is blind: decode runs as graph replays, launched
asynchronously, at one to a few requests per step, under a load the trace
paces. What matters, GPU time per step by layer and op across the whole run,
falls between CPU annotations that no longer run (inside graphs), CPU step
ranges the GPU work outlives, and windows that sample a load they do not
represent. One or two defects are inherent; three are missing features in
Kineto or vLLM's profiler integration; two are how we used it.

## 2. The defects and their root causes

| # | Defect | Measured | Root cause | Kind |
| --- | --- | --- | --- | --- |
| 1 | Call sites and shapes vanish in graph replay | every kernel of one replay shares the single `hipGraphLaunch` correlation id (Plan 11); shapes recorded for eager ops only, 19.8% of GEMM calls at C8 and 9.2% at C32 (Plan 13) | a replay runs no Python or ATen, so nothing on the CPU names or sizes its kernels; the names and shapes existed at capture time, and nothing joins them to replay | inherent to graphs, plus a **missing feature** (capture-to-replay join) |
| 2 | Many kernels, one name | every MXFP8 linear appears as `_mxfp8_linear_kernel.kd`; its projections cannot be separated (Plan 19) | kernels share a symbol; the trace keeps the symbol, not the call site or arguments | kernel naming, plus missing metadata |
| 3 | Steps miss their kernels | 0 of 10,874 `dsa_indexer_logits` kernels inside GPU `ProfilerStep#` spans; those spans start a median 23.8 ms after their CPU annotation; a line counter found 704 "steps" in a 200-step window; at C8 the window's span over its step count (35.1 ms) is 2.8× the median step (12.4 ms), and its 39.5% GPU-busy fraction mixes two populations: 193 steps run about 91% busy, 7 hold 91.6% of the idle (Plan 19) | steps are CPU ranges, and much of an asynchronously launched step's GPU work runs after its CPU range closes; the per-stream GPU projections cover only a small part of the work (449 µs and 7.9 µs of a ~12 ms step), because a kernel widens only the innermost user scope's span and most launch inside the nested `execute_…` scope (reasoned from Kineto's code); each step is one CPU event plus one per active stream under the same name; the mean period is inflated by a few long, mostly idle steps (one lasted 1.5 s with zero kernels) | **design choice**: no GPU-side step definition |
| 4 | Windows land in warmup | every C32 capture in Plans 10–15 profiled aiperf's warmup (1–2 running, not about 14) | the capture opens on a 420 s timer; the measured phase starts at 465–483 s | our usage (harness) |
| 5 | The profiler moves the result | a profiler-teardown block of about 19 s, 49.8 ms outside the excluded window, read as a stall (D04) | trace export runs synchronously in the serving process | partly inherent (overhead), mostly fixable (export) |
| 6 | Windows do not compose | the weighted cell model predicted the served step for only 1–3 of 16 cost centres within spread (Plan 19, D01) | short windows at controlled cells are samples of a mixed, trace-paced load | **inherent to windowed sampling** |
| 7 | Fragile artifacts | raw traces node-local and gone with the allocation (D01 found none under Plan 19's tree); a 700k-event trace cannot be parsed on a login node (Plan 11); pretty-printed Kineto JSON broke a line-oriented parser (D02) | Chrome JSON as the storage format; per-node files | format |
| 8 | Integration traps | vLLM's wrapper builds `torch.profiler.schedule` only when warmup or wait is non-zero, so `active_iterations` is silently ignored otherwise (Plan 19, `wrapper.py:218-226`); the profiler arms once per server lifetime from argv (D02); hooks inside PIECEWISE graphs never fire because a replay runs no Python (D02, 147 `site_unreached`) | vLLM profiler wiring; graphs again | **vLLM bugs** and graph semantics |

What no general tool can give:

- **Zero overhead.** It can be measured (a paired A/B, tracer on and off,
  same node) and made visible (the tracer's own intervals as events), not
  removed.
- **Cause from a trace alone.** A trace shows coincidence. D04 could say only
  "consistent with a profiler artifact"; a cause needs an intervention.

## 3. Fixes that belong upstream

- **vLLM**: build the schedule when warmup is 0; export asynchronously or out
  of process; allow more than one capture per server lifetime.
- **PyTorch / Kineto**: attach capture-time `record_function` scopes and
  shapes to kernels launched by a graph replay. On NVIDIA the platform
  already identifies graph nodes (Nsight Systems' node-level graph tracing,
  CUPTI's graph node ids). Whether ROCm's profiler SDK exposes the
  equivalent on MI355X is Plan 23's first question.
- **Kernel authors**: a distinct symbol per variant (AGENTS.md §10 already
  asks this of ours).

