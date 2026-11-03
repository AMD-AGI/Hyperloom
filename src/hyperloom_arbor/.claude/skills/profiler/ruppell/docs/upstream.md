# Upstreaming ruppell: what can go to PyTorch and the ROCm profiler, and what cannot

Written 2026-10-01 from the sources linked below. They move quickly (Kineto was
archived the day before); recheck them before acting on this.

## The short answer

Partly, and mostly by helping work that has already started rather than by
sending ruppell's code. ruppell exists because, on our pinned stack (ROCm
7.2.3, rocprofiler-sdk 1.3.2, PyTorch 2.12), nothing upstream attributes a
graph-replayed kernel to the code that captured it. Upstream is now closing that
gap from both ends:

- **rocprofiler-sdk** (develop docs, 1.4.x; ROCm 10.0.0 release notes) puts a
  `graph_exec_id` and a `graph_node_id` on every kernel-dispatch and memory-copy
  record from a graph launch. The node id is the dispatch's 0-based position
  within its `hipGraphLaunch`: the same join ruppell makes by position. It is
  stable across launches only under conditions the docs list (below).
  ([using rocprofv3](https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/develop/how-to/using-rocprofv3.html))
- **Kineto** froze on 2026-09-24 and was archived on 2026-09-30 while it is
  upstreamed into PyTorch ([pytorch/kineto](https://github.com/pytorch/kineto),
  migration in issue #1571). Its open PR
  [#1567](https://github.com/pytorch/kineto/pull/1567) attributes HIP graph
  kernels to graph nodes on ROCm by the same position rule, through the SDK's
  external-correlation request service. A reviewer asked for the fix to go into
  the SDK, or for the feature to be opt-in, because launches from four or more
  threads alias silently.
- **PyTorch** (2.15, nightly today) has capture-time kernel annotations:
  `torch.cuda.graph(..., enable_annotations=True)`, `mark_kernels(...)`, and
  `export_chrome_trace(cuda_graph_annotations=...)`. They need CUDA 13.1
  (`cudaGraphNodeGetToolsId`), are manual labels (no op names or shapes), and
  do not cover ROCm
  ([tutorial](https://docs.pytorch.org/tutorials/advanced/cuda_graph_annotations_tutorial.html)).

So the identity half of ruppell's join is arriving in the SDK, and the
annotation half is arriving in PyTorch, for CUDA and by hand. What ruppell can
add upstream is its evidence and the parts those efforts do not have yet:
automatic attribution (every op and module, with shapes), a check that refuses a
replay that does not match its capture, ROCm coverage, and GPU-anchored steps.
The serving-specific layers (engine steps, the workload schema, the census, the
gates) belong to vLLM and TraceLens, not to PyTorch or the profiler.

## Piece by piece

| ruppell piece | target | verdict | why |
| --- | --- | --- | --- |
| Graph identity on ROCm dispatch records | rocprofiler-sdk | **done upstream** (SDK develop, ROCm 10.0) | `graph_exec_id` / `graph_node_id`, plus a `graph_launch` record per launch with its `kernel_dispatch_count` (`--hip-graph-trace`). Nothing to send; two pieces of feedback below |
| Its stability conditions | rocprofiler-sdk | **can: feature request** | the node id names the same node only with segmented scheduling, `AMD_DIRECT_DISPATCH=1`, one launching thread and no `hipGraphExecUpdate`; packets written by another thread get zero ids, "indistinguishable from non-graph rows" (SDK docs). A per-dispatch flag for "graph launch, not attributed", or a per-launch check, would let a consumer refuse instead of guess |
| The mis-correlated replay (a graph's second-queue kernels tagged with a neighbouring `hipEventRecordWithFlags` call's id, 4 kernels once in 17,503 replays) | rocprofiler-sdk | **can: bug report** | evidence in [`milestones/m2.md`](milestones/m2.md) and `tests/m2/odd_replay.py` (job 48942, SDK 1.3.2). Reproduce on the SDK's develop branch first; it may be the same thread limitation as above |
| Capture-time attribution of replayed kernels: every captured launch tagged with the module and aten op on the stack, shapes included | PyTorch profiler | **can, partly: as an RFC and a prototype, not ruppell's code** | `mark_kernels` already joins capture-time labels to replayed kernels on CUDA. The extension is to record the `RecordFunction` stack (op name, input shapes, module) for every captured node when `enable_annotations=True`, so every kernel is attributed without manual labels, and to join on ROCm through the SDK's node id. ruppell shows the join holds at serving scale: 40,126 of 40,126 replays joined per rank, 99.99% of GPU time attributed ([`milestones/m2.md`](milestones/m2.md)). ruppell's own implementation lives outside PyTorch (its own SDK tool, module hooks, a separate `RecordFunction` library), so it is evidence and a reference, not a patch |
| Refusing a replay that does not match its capture (count and kernel sequence checked per replay) | PyTorch (#1567's successor) and the SDK | **can: review input and tests** | #1567 aliases under concurrent launches and its reviewer would "rather leave the information out than present it incorrectly". ruppell's rule does exactly that: attribute a replay only when it matches its capture, otherwise leave it unattributed and count it. Its probes (forked graph, two-queue graph, reused handles) can become tests |
| Steps on the GPU timeline (a marker kernel on the step's stream at each boundary) | PyTorch profiler | **partly: an opt-in RFC** | `ProfilerStep#` is a CPU range, and asynchronous GPU work outlives it (M03 §2, defect 3). An opt-in GPU-anchored step (a marker, or a timestamped event, on the current stream at `profiler.step()`) is generic. It adds work to the GPU stream, so it must stay opt-in. *Where* a step begins is the framework's: vLLM's engine step, not PyTorch's |
| Detail sampling (kernel records for K of every N steps, by step kind) | rocprofv3, PyTorch | **mostly exists; the missing part is not theirs** | rocprofv3 traces only between `roctxProfilerResume` and `roctxProfilerPause` with `--selected-regions` ([ROCTx](https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/develop/how-to/using-rocprofiler-sdk-roctx.html)); `torch.profiler.schedule(wait, active, repeat)` samples steps periodically. What ruppell adds is choosing steps by *kind* (every prefill and mixed step, decode at 5/1000), which only the server's scheduler knows: vLLM's profiler config, not PyTorch or rocprof |
| A CUDA finding from sampling: stopping CUPTI's kernel records while a step's kernels were in flight left their end times milliseconds late ([`milestones/cuda-c3.md`](milestones/cuda-c3.md)) | NVIDIA (CUPTI) | **can: bug report**, not to PyTorch or AMD | it is CUPTI's behaviour, seen on CUPTI API 130001 |
| The tuning pin (Inductor's and Triton's autotune choices recorded once and replayed in every rank and server) | PyTorch Inductor | **partly: evidence, not code** | the need is real: traced ranks ran grids 128 and 512 where the untraced server ran 256 (job 49826), and vLLM has the same report for another model ([vLLM #58636](https://github.com/vllm-project/vllm/issues/58636)). PyTorch is working on it: `TORCHINDUCTOR_DETERMINISTIC`, the remote autotune cache, and distributed runtime autotuning ([PR #176920](https://github.com/pytorch/pytorch/pull/176920)). ruppell patches private internals, which cannot go upstream; a "strict replay" use case (fail when a choice is not in the file) is worth adding to that discussion |
| The CPU op tree (aten ops from `RecordFunction`, outside Kineto) | PyTorch | **no need** | it reimplements what the PyTorch profiler already records; it exists only so the op tree can sit beside ruppell's own SDK tool. Its upstream value is the capture-time attribution above |
| The vLLM adapter (engine steps, scheduler fields, graph keys, named startup steps, signature checks) | vLLM | **not PyTorch or rocprof; partly vLLM** | vLLM owns the step, the captures and the load. Proposing stable hooks for step boundaries and graph capture, and the profiler fixes in the runner's M03 §3 (schedule built when warmup is 0, more than one capture per server, export off the serving path), would let any tool, ruppell or the PyTorch profiler, do this without wrapping vLLM's methods |
| The serving schema, the census, the stratified profile, phase coverage, collectives split into transfer and waiting | vLLM and TraceLens | **cannot go to PyTorch or rocprof** | they describe a workload (requests, rows, regimes) and compare ranks; a per-process profiler sees neither. The fields belong to the server, the analysis to TraceLens, which could take phase coverage and the transfer estimate (each launch's shortest duration across ranks) |
| Binary records, the offline join engine, the exports | none | **no** | rocpd is the SDK's own database now and carries the graph ids; ruppell should learn to read rocpd rather than offer its format |
| The repair of a mis-correlated replay; resolving reused graph handles in record order | none | **no** | workarounds. The repair goes away with the SDK fix; the handle problem with the SDK's process-monotonic `graph_exec_id` |
| The witness method and the gates (same GPU program, kernel-time parity, the operating point) | none as code | **method, not code** | no upstream place holds a validation protocol. It belongs in docs or a write-up, and in the tests of the pieces above |

## What to do, in order

1. **rocprofiler-sdk.** Reproduce the mis-correlated replay on the develop SDK
   and report it with ruppell's evidence. Ask for unattributed graph dispatches
   to be marked as such, and for a per-launch check (its dispatch count against
   the graph's, already in the `graph_launch` record) to be usable by consumers.
2. **PyTorch, ROCm graph attribution.** Kineto's PRs now go to PyTorch (#1571).
   Help carry #1567 across with a refusal rule and ruppell's test cases, built on
   the SDK's node id where the SDK has it.
3. **PyTorch, automatic capture-time attribution.** An RFC extending
   `enable_annotations` from manual labels to the op and module stack of every
   captured node, on CUDA and ROCm, with ruppell's M2 and M3 results as the case
   for it.
4. **PyTorch, GPU-anchored steps.** A small opt-in RFC.
5. **vLLM.** Step and capture hooks, and the profiler fixes.
6. **Inductor.** Add ruppell's evidence to the deterministic and distributed
   autotuning work.

## What this means for ruppell

On our image none of the upstream pieces is available yet: the graph ids need
the ROCm 10 SDK, and the annotations PyTorch 2.15 on CUDA 13.1. Until the image
moves, ruppell's runtime layer is still needed. Once the SDK's ids and
PyTorch's capture-time attribution reach the image, the runtime layer can shrink
to a reader of rocpd and of PyTorch's annotations. The vLLM adapter, the
serving schema, the census and the checks stay: they are about the server and
the workload, which is where upstream will not go.
