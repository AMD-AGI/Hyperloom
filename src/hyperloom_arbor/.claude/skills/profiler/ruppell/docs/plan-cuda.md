# CUDA plan: the runtime layer on B200 (CUPTI)

> Ports the runtime layer to NVIDIA B200 through CUPTI, keeping the Python
> API, the adapters, the schema and the census unchanged, and shows on a CUDA
> graph probe and a PyTorch demo what M1 showed on ROCm: every kernel,
> replayed or eager, carries its scope and its GPU-defined step, and an
> independent collector sees the same dispatches. The CUDA backend of M4
> ([`roadmap.md`](roadmap.md); design.md §6 W4). What is still open on ROCm:
> [`../TODOs.md`](../TODOs.md).

Status (2026-09-30): **C0–C4 run** on B200. Answers to C0:
[`feasibility-cuda.md`](feasibility-cuda.md) (node ids and capture callbacks
work as hoped; the witness cannot share the process on CUDA 13.0, so risk 1
resolved to the weaker witness). Records:
[`milestones/cuda-c3.md`](milestones/cuda-c3.md),
[`milestones/cuda-c4.md`](milestones/cuda-c4.md). The sections below are the
plan as written before C0.

## 1. Objective

At the end:

- `python -m ruppell run` on a B200 node loads a CUPTI tool library into the
  traced process and writes the same run directory as on ROCm: binary records
  for kernels, API calls and the graph lifecycle, kernel names, definitions,
  steps, clock pairs, the tool's own time, and `meta.json` with counts and
  drops.
- The C API the Python side calls (`ruppell_push`, `ruppell_pop`,
  `ruppell_define`, `ruppell_capture_key`, `ruppell_step_cpu`,
  `ruppell_step_fields`, `ruppell_flush`, `ruppell_marker_launch`) keeps its
  signatures, so `api.py`, `torch_hooks.py` and the vLLM adapter do not
  change.
- Replayed kernels join to their capture by graph node id where CUPTI reports
  it; a kernel whose node is not in the registry is refused with its reason,
  never placed by guess.
- An independent collector on the same work agrees per kernel, or, if that
  cannot run in the same process on the pinned stack, the weaker check that
  can is stated as such.

## 2. What changes, part by part

The platform layer is small: `runtime/src/tool.cpp` (677 lines) and
`runtime/src/marker.hip` (19 lines). Everything above it is platform-neutral
except where noted.

| Part | ROCm today | On CUDA | Size |
| --- | --- | --- | --- |
| Loading | `rocprofiler_configure` via `ROCP_TOOL_LIBRARIES` | `InitializeInjection` via `CUDA_INJECTION64_PATH`, set by `cli.py` | small |
| Kernel and API records | one lossless SDK buffer (kernel dispatch, HIP runtime API) | CUPTI activity buffers: `CONCURRENT_KERNEL`, `RUNTIME`, `DRIVER`, `EXTERNAL_CORRELATION`; drops from `cuptiActivityGetNumDroppedRecords` | a new `tool_cuda.cpp`, about tool.cpp's size |
| Scope on a kernel | external correlation on the dispatch record itself | `cuptiActivityPushExternalCorrelationId` emits separate records mapping correlation id to external id; written as their own file, joined in the engine | small, tool and engine |
| Kernel names | code-object callbacks | in the kernel record; the tool interns names and writes `kernels.tsv` | small |
| Graph lifecycle | HIP API callbacks: capture begin and end, instantiate, launch, launches while capturing | runtime API callbacks for the same calls, plus resource callbacks `GRAPHNODE_CREATED`, `GRAPHNODE_CLONED`, `GRAPHEXEC_CREATED` | medium |
| Clock pairs, periodic flush, `meta.json` | SDK timestamp; `rocprofiler_flush_buffer` | `cuptiGetTimestamp`; `cuptiActivityFlushAll` every `RUPPELL_FLUSH_S` and at exit | small |
| Marker | `marker.hip`, hipcc | `marker.cu`, nvcc for `sm_100` (CUDA 12.8 or later) | trivial |
| Version pin | ROCm and SDK version | CUDA toolkit, driver and CUPTI version (`cuptiGetVersion`), checked at load; unknown fails loudly | small |
| Build | `runtime/build.sh`, g++ and hipcc | a CUDA branch: g++ against CUPTI, nvcc | small |

## 3. Where CUDA is simpler: joining replays by node id

On ROCm the SDK gives a replayed kernel no graph node id, so the engine joins
a replay's dispatches to the capture by position (ordered by dispatch id),
refuses graphs whose replays differ in length or sequence, and since `014805d`
repairs replays whose kernels the SDK tagged with another call's correlation
id.

CUPTI's kernel record carries `graphId` and `graphNodeId` ("the unique ID of
the graph node that launched this kernel through graph launch APIs"). If
`GRAPHNODE_CREATED` fires inside each launch call made while a stream is
capturing, on the capturing thread, the tool records the active scope per
node at capture, and each replayed kernel joins by node id with no ordering
assumption. That is what design.md §4 already asks for: by node id where the
platform reports it, else by position.

Effect on open items in `TODOs.md`: graphs with parallel branches lose the
ordering problem; graph update after instantiation (`cudaGraphExecUpdate`)
is still not handled, because the registry holds the parameters seen at
capture.

## 4. Engine and analysis

`python/ruppell/engine.py` assumes ROCm in a few places:

- it orders a replay's dispatches by `dispatch_id`, and the repair uses it;
  CUPTI has no global enqueue sequence;
- queues and agents stand for streams and devices;
- API call names are `hip*`, and the Chrome export's category is `hip_api`.

Changes: record version 2 (`graph_id`, `graph_node_id` in the dispatch
record; a platform field in `meta.json`); the node-id join as a second path
beside the position join, which stays for ROCm; the external-correlation
join; platform-neutral names in the export. `validate/witness.py` gains a
reader for the CUDA witness (nsys, section 5).

## 5. Risks, most serious first

1. **The witness.** AGENTS.md requires an independent collector on the same
   work; on ROCm the strong form runs `rocprofv3` in the same processes.
   CUPTI has allowed one subscriber per process, so nsys, Kineto and ruppell
   exclude each other. CUPTI's documentation says CUDA 13.3 with driver r610
   or later supports several subscribers through `cuptiSubscribe_v2`, but
   only when every tool uses the v2 API. Either the pin is at or above that,
   with an nsys that uses v2, or the witness is weaker (separate runs,
   compared by counts per step, not per dispatch) and every result says so.
2. **What the graph ids mean.** The documentation does not say whether
   `graphNodeId` in a kernel record names the node of the captured graph or
   of the instantiated one, nor whether `GRAPHNODE_CREATED` fires during
   stream capture. If the record names the instantiated node, `GRAPHNODE_CLONED`
   must give the mapping back to the captured node. Only a probe answers
   this.
3. **`torch.compile` in vLLM on NVIDIA.** vLLM's default there compiles the
   model and captures piecewise and full graphs. Module forward hooks may not
   run inside compiled regions (a `TODOs.md` item that M2 avoided because the
   served ROCm configuration did not compile). On B200 it blocks scopes
   unless the served configuration turns compilation off; the fix (scopes
   from `RecordFunction` or from dynamo's module metadata) is shared with M5.
4. **Kineto in the same process.** PyTorch's profiler also takes CUPTI;
   anything that enables it (for example `KINETO_USE_DAEMON`) collides with
   ruppell under the one-subscriber limit. The tool detects a failed
   subscription and refuses to trace, with the other subscriber's name
   (`cuptiSubscribe_v2` reports it).
5. **Overhead.** Per-kernel activity records inside graph replays have a
   cost, and DSV4.1 on ROCm ran at about 96 k records per second. Measured
   with a paired A/B, tracer off and on, same node, before any claim.
6. **Access.** A B200 node, a pinned container, and a place to parse traces
   that is not a shared login host.

## 6. Work, in order

### C0 Facts first (a probe, no ruppell changes)

A small CUDA program: two streams, a captured graph with a fork, eager
launches, a memset node, each launch under a pushed external correlation id;
instantiate, replay 100 times. One job on the pinned stack answers, and
`feasibility-cuda.md` records with the job id:

1. Does `graphNodeId` in replayed kernel records name the captured or the
   instantiated node? Does `GRAPHNODE_CLONED` map one to the other?
2. Does `GRAPHNODE_CREATED` fire during stream capture, inside the launch
   call, on the capturing thread?
3. Which correlation and external ids do replayed kernels carry?
4. Can nsys and a second CUPTI subscriber trace the same process on the
   pinned CUDA and driver? If not, which witness is left?
5. Records per second and drops at the probe's rate; the activity buffer size
   that keeps drops at 0.

Stop here if 1 and 2 both fail: then CUDA needs the position join too, and
this plan is redone.

### C1 Runtime

`runtime/src/tool_cuda.cpp`, `runtime/src/marker.cu`, a CUDA branch in
`runtime/build.sh`, `CUDA_INJECTION64_PATH` in `cli.py`, the version pin.
Same files, same C API, same kill-robust flushing as on ROCm.

### C2 Engine

Record version 2, the node-id join, the external-correlation join,
platform-neutral names, the witness reader. Offline tests with recorded
fixtures from C0.

### C3 An M1 equivalent on B200

The probe and the PyTorch demo (`instrument()`, a captured graph, steps):
every kernel scoped and in its step, the Chrome trace loads in Perfetto and
`chrome://tracing`, the witness agrees, overhead measured with the paired
A/B. Recorded as a milestone record like `milestones/m1.md`.

### C4 vLLM on B200

The adapter unchanged; the served configuration's compile mode and graph
count recorded first. Depends on risk 3.

## 7. Effort (rough, before C0)

| Step | Effort |
| --- | --- |
| C0 probe | 1–2 days |
| C1 runtime | about a week |
| C2 engine | about a week |
| C3 M1 equivalent | about a week |
| C4 vLLM | 2–4 weeks, mostly risk 3 |

About 3–4 weeks to an M1-grade result on B200 if C0 comes back favourable.

## 8. References

- [CUpti_ActivityKernel9](https://docs.nvidia.com/cupti/api/structCUpti__ActivityKernel9.html): `graphNodeId`, `graphId`, `correlationId`, `streamId`
- [CUPTI callback API](https://docs.nvidia.com/cupti/api/group__CUPTI__CALLBACK__API.html): graph resource callbacks
- [CUPTI usage](https://docs.nvidia.com/cupti/main/main.html): multiple subscribers from CUDA 13.3 and driver r610
- [CUPTI result codes](https://docs.nvidia.com/cupti/api/group__CUPTI__RESULT__API.html): `CUPTI_ERROR_MULTIPLE_SUBSCRIBERS_NOT_SUPPORTED`
- [TensorRT-LLM #9073](https://github.com/NVIDIA/TensorRT-LLM/issues/9073): the subscriber conflict with nsys and Kineto on B200
- [PyTorch #112389](https://github.com/pytorch/pytorch/issues/112389): `KINETO_USE_DAEMON` taking CUPTI at import
