# Design: three layers

> Copied on 2026-09-29 from `jiaweic-runner`, `plans/arbor/M03_GRAPH_AWARE_TRACER.md`
> sections 4–7. From here on this file is the design of record; the runner's
> M03 keeps the analysis and points here.

## 4. The design: three layers, all ours, in one package

Operator, 2026-09-29: we build the runtime layer ourselves rather than lean on
Kineto or `rocprofv3` for collection, for control and for update speed, and
accept maintaining one more package. Everything lives in
[`the ruppell repository`](ruppell's repository (AMD internal))
(private). It was created as `<fork>/ruppell` on 2026-09-29, empty, and moved
with its history on 2026-10-19.

### 4.1 What "ourselves" means

We own everything above the platform's documented tracing interface: the tool
library, the graph registry, the step markers, the buffers, the storage
format, the join, the export and the validation. We do not reimplement how
timestamps come off the GPU. On ROCm that interface is the ROCm profiler SDK
(`rocprofiler-sdk`), loaded as a tool library the way `rocprofv3` itself is;
on CUDA it would be CUPTI. Below that is driver territory, which would add
code without adding control.

What we stop depending on: `torch.profiler` and Kineto, vLLM's profiler wiring
(and its one-capture-per-server and schedule traps), and the `rocprofv3` CLI
and its formats. They remain useful as **independent witnesses** that
`ruppell`'s numbers are checked against.

### 4.2 The three layers

| Layer | Written | Contents | Fixes (§2) |
| --- | --- | --- | --- |
| **Runtime** (C++ tool library plus a thin Python binding) | once per platform: ROCm first, CUDA later | kernel dispatch records with GPU start and end, correlation ids and dispatch parameters; HIP API records for graph capture (`hipStreamBeginCapture`, `hipStreamEndCapture`, `hipGraphInstantiate`) and replay (`hipGraphLaunch`); a **graph registry** built at capture from the launches recorded while each graph was captured, joined at replay by graph node id where the SDK reports it, else by position in the replay; **step markers** (a one-thread marker kernel carrying the step id, enqueued on the stream at every boundary, so steps exist on the GPU timeline); per-thread lock-free ring buffers drained by a background thread to compact binary files; CPU and GPU clock pairs at start and periodically; the tool's own intervals recorded as events | 1, 2, 3, 5, 7 |
| **Framework adapter** | once per framework: vLLM first | a **capture scope** around each graph capture (graph key, and the op and module scope stack at each launch, from a scope API or PyTorch's `RecordFunction` callbacks, which carry op names and input shapes without running `torch.profiler`); `step_begin` and `step_end` around every engine step; the scheduler's per-step state handed to the schema layer. Behind one env flag, default off | 1, 2, 3, 8 |
| **Workload schema** | per workload type: serving first | a declaration of the fields that define a step's load and a run's regime (serving: running requests, rows, context lengths, prefill or decode, graph key or eager, the client's phase clock; training later: global step, micro-batch, pipeline stage), how each is aligned in time, and which are required | 4, 6 |

On top of the three layers, in the same package:

- **Census**: per-step aggregates (GPU time by registry key) for every step of
  a run, and full kernel detail on a declared sample of steps.
- **Export**: Perfetto and Chrome JSON produced from the binary files on the
  node, so TraceLens and the Perfetto UI keep working.
- **Validate**: every output states attributed share of GPU time, unattributed
  kernels by name, registry-versus-replay sequence agreement, and per-step
  kernel sum against the step span; plus a check against an independent
  witness (`rocprofv3` or Kineto) on the same run.

### 4.3 Package layout (proposed)

```text
ruppell/
  runtime/           C++ tool library: SDK backend (ROCm), buffers, registry, markers, clocks
  python/ruppell/    binding and API: init, scope, capture, step, context
  adapters/vllm/     capture scope, step hooks, scheduler context (behind one flag)
  schemas/           serving.yaml (training.yaml later)
  census/  export/  validate/
  tests/             standalone graphs with known kernels, golden traces, overhead microbenchmarks
```

### 4.4 The cost of owning it

- **Platform drift.** The SDK's interface moves with ROCm releases. `ruppell`
  pins the ROCm versions it supports, runs its standalone tests on each image
  it is used with, and fails loudly on an unknown version.
- **Framework drift.** An adapter follows its framework's internals (where
  vLLM captures graphs and steps its engine). Keep adapters thin and propose
  their hooks upstream, so they stop being patches.
- **Testing needs a GPU.** The standalone tests run on a node; their results
  are part of each release's record.
- **Correctness is ours to prove.** Every release is checked against an
  independent witness before it is used for a claim.

Pieces of the runtime and adapter layers are generic to any graph-captured
serving stack; nothing in them is specific to DeepSeek V4.1 Flash.

## 5. How it fits the other notes

- **M02**: `ruppell` is the engine behind the L3 skills `profile-capture`,
  `kernel-census`, `step-attrib` and `validate-profile` (M02 §5.3); its census
  feeds the knowledge base's attribution step (M02 §7). It is a package the
  skills call, versioned and pinned like any other.
- **M01**: a profile informs directions and never gates (AGENTS.md §9.2, Plan
  18). `ruppell`'s validation outputs are evidence, not gates.
- **S01**: parts 6 and 7 of the pre-flight (profiling plan and results) use it.
- **The runner**: `ruppell` is pinned in `sources.lock.json` and checked out as
  a submodule, like vLLM and Arbor.

## 6. Candidate workstreams

| W | Workstream | Depends on |
| --- | --- | --- |
| W1 | All three layers, ROCm and vLLM and serving, validated on DSV4.1 Flash (Plan 23) | none |
| W2 | Upstream asks: the vLLM fixes of §3, and the adapter's hooks, as PRs | W1 |
| W3 | A second framework adapter (SGLang) and a second model, to test the layer split | W1 |
| W4 | CUDA backend (CUPTI) for the runtime layer | W1 |
| W5 | A training schema and adapter | W3 |
| W6 | Package as Arbor skills with evaluations (M02) | W1, M02 W1 |

## 7. Open questions

- Does the ROCm profiler SDK on this image give dispatch records for kernels
  launched by `hipGraphLaunch`, with correlation to the launch, and graph node
  ids? Plan 23's 4.0 answers it. Without dispatch records for graph kernels,
  the runtime layer needs a deeper mechanism and the plan stops to decide.
- Scope source at capture: our own scope API, PyTorch `RecordFunction`
  callbacks, or both? The callbacks give op names and shapes for free, but tie
  the adapter to PyTorch.
- The binary format: Perfetto protobuf directly, or our own records converted
  offline?
- How much of the step is host-side (scheduling, sampling) that a GPU-anchored
  view shows only as idle gaps? Plan 16's scheduling-bubble work is the
  reference; CPU-side intervals may need their own records.
- Where the always-on census lives after a run: the knowledge base, the
  results registry, or both.
