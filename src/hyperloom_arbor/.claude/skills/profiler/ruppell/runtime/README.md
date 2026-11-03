# runtime: the platform layer

A C++ tool library loaded into the traced process by the ROCm profiler SDK
(`src/tool.cpp`), a HIP marker library (`src/marker.hip`), and the record
layouts they share with the Python side (`src/records.h`). On CUDA, the twins
`src/tool_cuda.cpp` (a CUPTI injection library) and `src/marker.cu` (see "On
CUDA" below). Build: `bash build.sh` (g++ for the tool, hipcc or nvcc for the
marker and the probe; `RUPPELL_PLATFORM` picks rocm or cuda, by default cuda
when nvcc is found and hipcc is not); `RUPPELL_BUILD` sets the output
directory.

## What it records

- **Kernel dispatches** (lossless SDK buffer): GPU start and end, correlation
  to the launching call, external correlation (scope or marker), kernel id,
  dispatch id, queue, agent, grid and workgroup (`dispatch.bin`).
- **Memory copies on the DMA engines** (the SDK's memory-copy tracing, in the
  dispatch context, so sampled with kernel detail): start and end, bytes,
  direction, source and destination agents, correlation and external
  correlation (`copies.bin`, `ruppell_copy_rec`). `hipMemset` and the copies
  HIP runs as blit kernels are kernel dispatches. `RUPPELL_COPIES=0` turns
  copy records off.
- **ROCTX ranges** (`RUPPELL_NVTX=1`; off by default): the SDK's marker
  callbacks (push, pop, start, stop, mark), timestamped by the tool on the
  calling thread, written when each range closes (`ranges.bin`,
  `ruppell_range_rec`; names in `ranges.tsv`, domain `roctx`). PyTorch's
  `torch.cuda.nvtx` and `emit_nvtx` call ROCTX on ROCm.
- **HIP API calls** (same buffer): start, end, correlation, thread, operation
  (`api.bin`, names in `api_ops.tsv`).
- **Graph lifecycle** from HIP API callbacks: capture begin and end (with the
  graph), instantiate (exec to graph), launch (exec and its correlation id),
  and every launch or memory operation made while a stream is capturing, with
  the scope active at that moment (`capture.bin`).
- **Kernel names** from code-object callbacks (`kernels.tsv`).
- **Scopes, graph keys, context** defined by the Python API (`defs.tsv`) and
  per-step fields (`steps.tsv`).
- **Clock pairs**: SDK timestamp, realtime and monotonic clocks, every second
  (`clock.tsv`).
- **Its own time** in buffer callbacks (`self.bin`), and counts and drops in
  `meta.json`.

## How attribution works (python/ruppell/validate/core.py)

- Replayed kernels: the SDK gives them the `hipGraphLaunch` call's correlation
  id and no graph node id, so they join to the capture by position in the
  replay (ordered by dispatch id), only when every replay has as many
  dispatches as the capture has launches and runs the same kernel sequence;
  otherwise the graph is refused, with the reason.
- Eager kernels: the scope pushed on the launching thread (external
  correlation).
- Steps: intervals between marker kernels (`ruppell_step_marker`) on the GPU
  timeline, identified by the external correlation value pushed around each
  marker launch.

## On CUDA

`tool_cuda.cpp` is loaded by the CUDA driver at `cuInit`, through
`CUDA_INJECTION64_PATH` (set by `ruppell run`; `ruppell.init()` calls `cuInit`
so the tool is up before anything asks). It exports the same C API. It
records (record version 2):

- **Kernels** (CUPTI activity buffers, `CONCURRENT_KERNEL`): start and end,
  correlation id, kernel name (interned into `kernels.tsv`), stream, device,
  grid and block, and for a graph-launched kernel its **graph id and the
  instantiated graph node** (`dispatch.bin`, `ruppell_kernel_rec`).
- **Memory copies and sets** (`MEMCPY`, `MEMCPY2` for peer copies, `MEMSET`),
  switched with kernel detail: start and end, bytes, copy kind, stream,
  devices, and for a graph's memory node its graph and instantiated node
  (`copies.bin`, `ruppell_copy_rec`). `RUPPELL_COPIES=0` turns them off.
- **NVTX ranges** (`RUPPELL_NVTX=1`; off by default): CUPTI's NVTX callbacks.
  NVTX loads one injection library at the first NVTX call; `ruppell run` sets
  `NVTX_INJECTION64_PATH` to the tool, whose `InitializeInjectionNvtx2` hands
  NVTX on to CUPTI's. Push and pop per thread and per domain, start and end,
  marks, domain names and registered strings, payloads and categories
  (`ranges.bin`, `ruppell_range_rec`; names in `ranges.tsv`).
  `RUPPELL_NVTX_DOMAINS` keeps only the domains named. Under another
  injection (nsys) no range is recorded, and `meta.json` says "not injected".
- **Runtime and driver API calls** (`api.bin`; driver operations are
  `0x10000 | cbid` in `api_ops.tsv`) and **external correlation** records,
  correlation id to the scope or marker code pushed on the launching thread
  (`ext.bin`, kind `CUSTOM2`); the engine joins them by correlation id.
- **Graph nodes** from CUPTI resource callbacks: every node created, with the
  scope on top of the creating thread's stack (during stream capture that is
  the launch's scope), and every clone made at instantiation, clone to
  original; node parameter changes after capture (`capture.bin`, kinds 7-9).
  Capture begin and end and graph launches come from runtime and driver API
  callbacks; nothing else is intercepted.

Replayed kernels join by node id: instantiated node, through the clone
records, to the node created during capture. A replay joins whole or is
refused with the reason. The CUPTI API version is pinned (130001); another
version, or a CUPTI already held by another tool (nsys, Kineto), leaves the
process untraced and says why. What CUPTI reports and what it costs:
[`../docs/feasibility-cuda.md`](../docs/feasibility-cuda.md).

## Things the SDK taught us

See [`../docs/feasibility.md`](../docs/feasibility.md): globals must outlive
static destructors (the SDK finalizes later), kernels on one queue overlap by
up to a few hundred ns, replays start before `hipGraphLaunch` returns, and two
tools can share a process when both are preloaded.
