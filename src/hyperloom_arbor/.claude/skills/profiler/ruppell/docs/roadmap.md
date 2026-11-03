# Roadmap

Milestones toward the v0 plan ([`plan-v0.md`](plan-v0.md)). Each has exit
criteria that can be checked, and says which plan items it covers. Milestones
are ordered; steps inside a milestone are ordered unless marked parallel.

| Milestone | What exists at the end | Plan items |
| --- | --- | --- |
| **M0** scaffold | layout, contracts, docs | none (done: `f3f63e3`) |
| **M1** a working Chrome trace generator | `ruppell` traces a graph-captured PyTorch program on ROCm and writes a Chrome trace in which every replayed kernel carries its scope and its GPU-defined step | 4.0, 4.1, part of 4.4 (**done** 2026-09-29, `9322623`: [`milestones/m1.md`](milestones/m1.md), [`feasibility.md`](feasibility.md)) |
| **M2** vLLM on DSV4.1 Flash | the vLLM adapter and serving schema; a Chrome trace and per-step census of a short AgentX replay | 4.2, 4.3, 4.4 (plan: [`plan-m2.md`](plan-m2.md); **done for vLLM** 2026-09-30: [`milestones/m2.md`](milestones/m2.md), [`feasibility-m2.md`](feasibility-m2.md); other frameworks: [`adapting-frameworks.md`](adapting-frameworks.md)) |
| **M3** v0 validated | the plan's gates: coverage, agreement, overhead, correctness at C8 and C32 | 4.5, 4.6, 4.7 |
| **M4** beyond one stack | CUDA backend, a second framework (SGLang), a training schema | design.md §6 W3–W5 (CUDA on B200: [`plan-cuda.md`](plan-cuda.md)) |
| **M5** open items and the CPU tree | every item in [`TODOs.md`](../TODOs.md) closed by a job or test, or kept with a reason; a default-off CPU op and module tree in the Chrome trace | none (TODOs.md; CPU op tree built and shown on B200: [`cpu-ops.md`](cpu-ops.md)) |

After M5: [`roadmap-v2.md`](roadmap-v2.md), M6–M12, the gaps to Nsight Systems (copies, NVTX and
ROCTX, collectives, counters, CUDA breadth, a viewer, external control and many nodes), M13,
ruppell's own cost, and M14, components profiled in depth by default.

## M1: a working Chrome trace generator

> **Done** on 2026-09-29 (Cluster A job 47944, commit `9322623`): every point of
> "What working means" below holds for a HIP probe and a PyTorch demo; the
> record, and how the build differs from this sketch (a build script instead
> of CMake, position join, an in-process witness, overlap lanes), is in
> [`milestones/m1.md`](milestones/m1.md). The steps below are kept as they
> were planned.

### What "working" means

On a small PyTorch program that captures a HIP graph and replays it (a few
scoped linear, activation and norm layers, captured with
`torch.cuda.CUDAGraph`, which PyTorch uses for HIP graphs on ROCm, and also run
eagerly on a second stream), `ruppell` produces one Chrome trace JSON file that:

- opens in the Perfetto UI and in `chrome://tracing`;
- shows every kernel dispatch as a slice on its GPU queue's track, with the
  scope that launched it (the module path and op recorded at capture) in its
  name or arguments, for eager and replayed kernels alike;
- shows steps as slices on a GPU-side step track, bounded by marker kernels,
  and puts every kernel inside exactly one step;
- links each `hipGraphLaunch` on the CPU to the kernels it produced (flow
  events);
- carries a validation block: attributed share (100% on this program), kernels
  that could not be attributed (none), dispatch drops (zero);
- matches `rocprofv3` on the same program: identical dispatch counts, GPU time
  per kernel within 3%.

Not in M1: vLLM, the serving schema, the always-on census, the out-of-process
drain (an in-process background thread is enough here), overhead A/B, CUDA.

### Where it is built

On a ROCm GPU node, inside a container with ROCm, the ROCm profiler SDK and
PyTorch: the runner's `dsv41flash_arbor` image qualifies. The Windows and WSL
hosts have no ROCm; they only edit and push. Standalone programs use GPUs 4–7
while a node's GPUs 0–3 serve, and never run beside a measurement.

### Steps

**M1.1 Feasibility probes (plan 4.0).** Small programs in `tests/probes/`,
each writing a result file:
- the image's ROCm version; SDK headers, library and CMake package present;
- a minimal tool library that the SDK loads (confirm the loading mechanism,
  for example the tool-library environment variable, on this image) and that
  receives one callback;
- dispatch records for a directly launched kernel and for kernels launched by
  `hipGraphLaunch`: timestamps, correlation, and any graph or node id;
- HIP API callbacks for stream capture, instantiate and launch, and for kernel
  launches made while capturing;
- a graph's replay sequence identical over 1,000 replays;
- a one-thread marker kernel visible in dispatch records, in order;
- per-dispatch cost of the minimal tool;
- whether `rocprofv3` and `ruppell` can load in one process.

Exit: `docs/feasibility.md` with each answer and its evidence; the join mode
(node id or position) recorded. If graph-launched kernels produce no dispatch
records, M1 stops here (plan §6).

**M1.2 Build system.** `CMakeLists.txt` for `runtime/`, finding the SDK; a
`ruppell` Python entry that locates the built library; a make target that
builds and runs the probes. Exit: a clean build in the container from a fresh
clone, with the library digest printed.

**M1.3 Dispatch recording.** Dispatch records into per-thread ring buffers
with a drop counter; a background thread drains them to a binary file; clock
pairs at start and every second. A Python reader for the binary format.
Exit: a plain HIP program's kernels read back from the file, with no drops.

**M1.4 Chrome exporter (first trace).** `ruppell export RUN_DIR -o trace.json`:
- GPU kernels as complete events (`"ph": "X"`) on one track per queue,
  timestamps in microseconds on one aligned timebase;
- HIP API calls on per-thread CPU tracks;
- process and thread names as metadata events;
- flow events from each launch call to its kernels;
- the validation block in the file's metadata.

Exit: the plain HIP program's trace opens in both viewers, and its kernel
count equals `rocprofv3`'s.

**M1.5 Graph registry and scopes.** `ruppell.capture(key)` and
`ruppell.scope(name, ...)` implemented: launches recorded while capturing are
stored per graph with their scope; replayed dispatches join to them by the
mode M1.1 chose, after checking that the replay sequence matches, and refuse
otherwise. Exit: tests (a) and (b) pass (`tests/README.md`); in the trace,
replayed kernels carry their scopes.

**M1.6 Step markers.** `ruppell.step_begin` and `step_end` enqueue marker
kernels; the exporter draws a step track from them and tags each kernel with
its step. Exit: test (c) passes; step counts come from distinct step ids.

**M1.7 The PyTorch demo and the witness.** `examples/torch_graph_demo.py`
(scoped modules, capture, eager work on a second stream, steps around
replays); `ruppell run -- python examples/torch_graph_demo.py` produces the
run directory. Test (d) (zero drops at the highest dispatch rate M1.1 measured)
and test (e) (counts equal to `rocprofv3`, time within 3%) pass.
Exit: every point of "What working means" holds, recorded in
`docs/milestones/m1.md` with the `ruppell` commit and library digest.

### Risks

- **The SDK may not report graph-launched dispatches with enough to join.**
  Then position join depends on correlation to the launch and on stable replay
  order; if neither holds, M1 stops at M1.1 (plan §6).
- **Loading two tools at once** may not work; then the witness runs on a
  separate execution of the same program (plan 4.6 allows this).
- **Timebase alignment** between CPU API records and GPU dispatch records; the
  exporter refuses a trace whose clock pairs disagree beyond a stated bound.
- **Chrome JSON size**: fine for M1's small program; M2 adds sampling and
  Perfetto protobuf output.

## After M1

- **M2**: the vLLM adapter (capture scope, step hooks, scheduler state; plugin
  or flagged commits), the serving schema, per-step census and Chrome export on
  a short seed-42 AgentX replay of DSV4.1 Flash.
- **M3**: the plan's validation, overhead and correctness items, on runner
  nodes, settled against its gates.
- **M4**: CUDA backend, SGLang adapter, training schema.
- **M5**: work through [`TODOs.md`](../TODOs.md): show the unverified items
  work (compiled and graph-tree captures, parallel branches, memory nodes,
  graph update, MI355X, clock alignment across nodes, TraceLens, a person
  reading the traces), or close them with the reason they do not apply; build
  what is not implemented. And a **CPU op and module tree**, off by default:
  scope push and pop timestamped by the tool, exported as nested slices on
  each host thread (module path, op, shape) like the PyTorch profiler's CPU
  side, with launch flows from the innermost scope. Its cost is measured with
  a paired A/B, flag off and on, before it is used for a claim.
