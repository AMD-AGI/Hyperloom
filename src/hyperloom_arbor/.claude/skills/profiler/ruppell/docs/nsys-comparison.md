# ruppell and Nsight Systems: feature gaps (2026-10-20)

**What is compared:** ruppell at `ad14914` (`main`), against Nsight Systems as NVIDIA documents it
for 2026.5 (the sources at the end).
- **Our measurements:** ruppell's B200 work used nsys 2025.3.2 and 2026.3.2:
  [`milestones/cuda-c3.md`](milestones/cuda-c3.md),
  [`milestones/accuracy-study.md`](milestones/accuracy-study.md).
- **What is unconfirmed:** a feature marked "documented" here was read in NVIDIA's documentation,
  not tried by us.

## In short

**The two tools aim differently:**
- **nsys** records a whole system's activity: GPU work of every kind, CPU threads and samples,
  libraries, the network and hardware metrics. You read it in a timeline GUI, and in reports over
  a capture window.
- **ruppell** records less: GPU kernels and the API calls that launched them. Every kernel,
  replayed from a graph or not, gets attributed to the code that launched it (module path, op,
  shape) and to a step defined on the GPU timeline, with the workload's state per step. It checks
  its own records against the vendor's collector, and runs the same way on AMD and NVIDIA.

**What ruppell lacks that matters most for its use,** profiling LLM serving and training:

1. **Memory copies and fills.** Closed by M6 (2026-10-20): ruppell now records copies and sets
   on both platforms (`copies.bin`). Unified-memory migrations remain nsys's.
2. **Collective semantics.** Closed by M8 (2026-10-21): each collective kernel carries its
   operation, dtype, size and group, from vLLM's adapter or NCCL's and RCCL's own record. nsys's
   straggler and overlap recipes remain ahead of birdeye's tables.
3. **Hardware metrics.** nsys samples SM, tensor-core, DRAM, PCIe and NVLink activity up to
   200 kHz (`--gpu-metrics-devices`). ruppell has no counters at all.
4. **The CPU side.** ruppell's view of the host is its scopes, the aten ops (with the CPU op
   tree), API call intervals and step windows. nsys samples the CPU, shows thread states, traces
   libc and pthread calls, and samples Python stacks.
5. **Existing annotations.** Closed by M7 (2026-10-20): with `RUPPELL_NVTX=1` ruppell records NVTX and
   ROCTX ranges (`ranges.bin`), at about 0.5 µs per NVTX call on B200 and 4 µs per ROCTX call on
   MI350X.
6. **Hardware timestamps on Blackwell.** nsys uses them by default. ruppell's software timestamps
   read 0.2–0.5 µs longer per kernel (the `software-timestamps` limitation).
7. **A viewer and an analysis library.** nsys has a GUI timeline over whole reports, about 40
   `stats` reports, `analyze` rules and some 40 multi-report recipes. ruppell exports windows to
   Perfetto and TraceLens and writes birdeye's tables; its own viewer is on an unmerged branch
   (`viewer`).

**Where ruppell goes further** (last section):
- graph-replayed kernels attributed without annotating the program;
- steps carrying the scheduler's state;
- whole-run analysis with per-step sampling;
- AMD support;
- a built-in check against the vendor's collector;
- records that survive a killed process.

## Feature by feature

Priority is for ruppell's use: **high** where a gap can make its numbers wrong or leave a common
question unanswered, **low** where its workloads rarely need the feature.

### GPU activity

| Feature | nsys | ruppell | Gap |
| --- | --- | --- | --- |
| Kernels | `-t cuda`, every kernel | every kernel, or sampled steps (`RUPPELL_DETAIL`) | none |
| memcpy and memset | traced, per stream; unified-memory transfers too | since M6 (2026-10-20): every copy and set on CUDA, the DMA engines' copies on ROCm (fills there are kernels), with scope and step (`copies.bin`) | none; unified-memory transfers remain nsys's |
| Unified-memory page faults | `--cuda-um-cpu-page-faults`, `--cuda-um-gpu-page-faults` (up to 70% overhead) | no | low |
| GPU memory in use | `--cuda-memory-usage` | no | medium: KV-cache and allocator questions |
| CUDA events, cross-stream dependencies | `--cuda-event-trace` (driver ≥12.8) | no | low |
| GPU context switches | `--gpuctxsw` | no | low |
| Timestamps | hardware trace by default from Blackwell on; `-t cuda-sw` for software | software (CUPTI); on ROCm the profiler SDK's | **high** on Blackwell for short kernels: +0.2–0.5 µs each |
| Cost per traced kernel | on our probe the same as ruppell's (software); lower with hardware trace | ~0.7 µs (CUDA probe), ~2 µs inside graph launches (ROCm) | ruppell compensates by sampling |

### CUDA graphs

| Feature | nsys | ruppell | Gap |
| --- | --- | --- | --- |
| Replayed kernels | `--cuda-graph-trace=graph` (default, one item per graph) or `node` (per node, more overhead) | always per kernel, joined to the capture: by node id on CUDA, by position on ROCm | none |
| What code a replayed kernel belongs to | documented: NVTX ranges open while the graph was built are projected onto it (`nvtx-live`, the default; `nvtx-precapture`, experimental, for graphs built before collection). Seen in M7 (2026.3.2, `--cuda-graph-trace=node`): the `nvtx_gpu_proj_sum` report counted each replay under the range around its launch, and nothing under the ranges open at capture | module path, op and shape from its hooks and adapters; the aten op with the CPU op tree; since M7, the NVTX or ROCTX ranges open at capture | none for ranges; untried by us: nsys's `--pytorch=autograd-shapes-nvtx` on vLLM |
| Device-launched graphs | graph level (driver ≥12.3); node level with hardware trace and driver ≥13.0 | refused, with the reason | medium |
| Conditional and child-graph nodes, graph updates | traced | refused with the reason on CUDA; untested on ROCm | medium, as vLLM adopts them |

### Libraries, annotations and communication

| Feature | nsys | ruppell | Gap |
| --- | --- | --- | --- |
| User annotations | NVTX v3: ranges, marks, domains, payloads, counters | its own API (`scope`, `capture`, `step_begin`/`step_end`); since M7, NVTX and ROCTX push/pop and start/end ranges, marks, domains, registered strings and payloads (raw), with `RUPPELL_NVTX=1` | low: no NVTX counters, extension payloads or start/end projection; a ROCTX call costs about 4 µs |
| PyTorch | `--pytorch=autograd-nvtx\|autograd-shapes-nvtx\|functions-trace\|functions-trace-shapes` | module hooks; the CPU op tree (`RUPPELL_CPU_OPS`) for aten ops with dims and dtypes | close: both give ops and shapes |
| NCCL / RCCL | `-t nccl`: API, groups, kernels, P2P, proxy ops; recipes `nccl_sum`, `nccl_straggler`, `nccl_gpu_overlap_trace` | since M8 (2026-10-21): each collective kernel with its operation, dtype, size and group, from vLLM's adapter (its own kernels, replays included) or from NCCL's and RCCL's record through ruppell's profiler plugin (algorithm, protocol, channels); transfer estimate (shortest duration across ranks) and bus bandwidth per call kind (birdeye `coll_by_op`); start skew from clock-aligned starts | low: no proxy ops or per-peer P2P; no straggler recipe per operation; RCCL's MSCCL kernels carry no record |
| cuBLAS, cuDNN, cuSOLVER, cuSPARSE | `-t cublas,cudnn,...` | kernels by name only | low |
| MPI, UCX, OpenSHMEM, NVSHMEM, OpenMP, OpenACC | traced | no | low |
| Graphics, video, storage APIs (Vulkan, OpenGL, DX, NVENC/NVDEC, cuFile, S3) | traced | no | not needed |

### CPU side

| Feature | nsys | ruppell | Gap |
| --- | --- | --- | --- |
| CPU sampling with backtraces | `--sample`, `--backtrace=fp\|lbr\|dwarf`; Top-Down and Bottom-Up | no | medium: host-bound steps are diagnosed from API intervals and ops only |
| Python | `--python-sampling` (to 2 kHz), `--python-backtrace=cuda`, `-t python-gil` | Python scopes and the CPU op tree | medium |
| OS runtime and syscalls | `-t osrt` (libc and pthread calls), `--syscall` (beta) | no | low |
| Thread states, context switches | `--cpuctxsw` | no | medium |
| CPU counters | `--cpu-metrics` (full on Grace) | no | low |
| CUDA API backtraces | `--cudabacktrace` | the scope on top of the launching thread | low |

### Hardware metrics

| Feature | nsys | ruppell | Gap |
| --- | --- | --- | --- |
| GPU metrics | `--gpu-metrics-devices`, 10 Hz–200 kHz: SMs active, SM issue, tensor active, warp occupancy, DRAM, PCIe, NVLink (Turing+) | none ("No hardware counters") | **high** for "why is it slow"; on ROCm, rocprofv3's counters fill it per kernel |
| Power, temperature, clocks | NVML and DCGM plugins | no | low |
| NIC, switch and storage metrics | `--nic-metrics`, InfiniBand and Spectrum-X, `--storage-metrics` | no | low for one node; medium for multi-node serving |

### Collection control

| Feature | nsys | ruppell | Gap |
| --- | --- | --- | --- |
| What is captured | capture ranges (`cudaProfilerApi`, an NVTX range, a hotkey), `--delay`, `--duration`, repeated ranges | the whole run from launch; kernel detail on chosen steps (`RUPPELL_DETAIL`, `RUPPELL_DETAIL_KINDS`, `ruppell.detail()`); windows chosen afterwards (`export --steps`) | medium: no start/stop from outside the process |
| Interactive sessions | `nsys launch`, `start`, `stop`, `sessions` | no | medium |
| Several processes and ranks | the process tree, or system-wide | every process that loads the runtime writes its own directory; ranks merged with step ids and fingerprints checked | none on one node |
| Several nodes | one report per rank; time sync by UTC (ms) or TSC; vClock plugin (experimental) | untested across nodes (clock pairs agree within 256 ns on one node) | medium |
| Data from a killed process | trace buffers flush at the end of collection; `--cuda-flush-interval` defaults to 0 (10 s advised for long runs) | flushes every 5 s (`RUPPELL_FLUSH_S`); a worker marks its records complete at shutdown | ruppell ahead: nsys lost the end of every vLLM run at `--shutdown-timeout 0` on our B200 |
| Two tools in one process | CUPTI takes one subscriber (CUDA 13.0, driver r580): nsys and ruppell cannot share a process | the same; on ROCm, rocprofv3 and ruppell share one | platform limit |

### Outputs and analysis

| Feature | nsys | ruppell | Gap |
| --- | --- | --- | --- |
| Formats | `.nsys-rep`; export to SQLite, Arrow, Parquet, HDF5, JSON lines | binary records per process; Chrome trace; PyTorch-profiler JSON for TraceLens; birdeye CSVs and `REPORT.md` | medium: no columnar export, though birdeye reads nsys SQLite |
| Summaries | `nsys stats`: about 40 reports (kernels, API, memory, NVTX projections, MPI, unified memory, OS runtime) | `census` (every step: period, busy, GPU time by op and scope, the workload's fields); birdeye (TraceLens's tables, steps, categories, collectives, GEMM) | close for kernels and steps; ranges per name with their GPU projection (the engine's `nvtx` report, M7); no memory or API reports |
| Expert rules | `nsys analyze`: synchronous memcpy and memset, sync APIs, GPU gaps, GPU utilization | birdeye's "what stands out" list | medium |
| Multi-report analysis | `nsys recipe`: some 40 recipes, local or on Dask, writing Parquet and notebooks (kernel histograms and pace, NCCL straggler, GPU utilization maps, `diff`) | `birdeye-compare`; `witness` against another collector | medium |
| Viewer | GUI timeline over whole reports, Events view, multi-report view, browser streaming | Perfetto or `chrome://tracing` over windows (a full run is too large for JSON); a tiled viewer on the unmerged `viewer` branch | **high** for exploration |
| Programmatic access | `nsys_recipe` (pandas), JupyterLab extension, plugin SDK | `ruppell.engine`, `ruppell.birdeye.readers` (numpy) | close |

### Platforms

| | nsys | ruppell |
| --- | --- | --- |
| GPUs | NVIDIA, Turing and later | AMD MI350X and MI355X (ROCm 7.0.2 to 7.2.3); NVIDIA B200 |
| CUDA | 10.0 and later | CUPTI API 130001 only (CUDA 13.0); others refuse to trace |
| AMD | no | yes, the primary platform |
| OS | Linux x86_64 and Arm, Windows; embedded editions | Linux |

## Where ruppell goes further

- **Replayed kernels attributed without annotating the program.** Module path, op and shape come
  from module hooks and framework adapters (vLLM, DLRM), and aten ops from the CPU op tree. A
  graph that cannot be joined exactly is refused with the reason, never guessed. nsys gets there
  only through NVTX that the program emits during capture.
- **Steps on the GPU timeline, with the workload's state.**
  - Marker kernels bound each engine step, and every kernel belongs to exactly one, however far
    the CPU runs ahead.
  - The serving schema adds running requests, rows, context lengths, the step's kind and a
    fingerprint per step, checked across ranks.
  - nsys's steps are whatever NVTX ranges the program emits on the CPU. It projects each one onto
    the GPU work launched inside it (`nvtx_gpu_proj_*`), without the scheduler's state.
- **The whole run, sampled.** `RUPPELL_DETAIL=K/N` traces K of every N steps through the run, at
  under 1% throughput cost on B200. On MI350 the traced server held its operating point at C8,
  and at C32 all but requests in flight, 0.2% over the bound ([`milestones/m3.md`](milestones/m3.md)). The census covers
  every step. nsys records one window, or several repeated ranges, in full detail.
- **AMD.** One tool, one set of records and reports, on ROCm and CUDA.
- **A check against the vendor.** `ruppell witness` compares its records with rocprofv3 (same
  process, dispatch by dispatch) or nsys (separate runs). The limitations that apply to a run
  travel inside every output.
- **Robust to how servers die.** Periodic flushes and a completion mark per worker, so a server
  killed at shutdown still yields its records, and the analysis says which ranks ended early.

## What to do about the gaps

Planned as milestones M6–M12 in [`roadmap-v2.md`](roadmap-v2.md) (M13 there is ruppell's own cost, M14 components profiled in depth). In the order that pays most for
ruppell's use:

1. **Record memory copies and fills.** Done (M6, 2026-10-20): CUPTI's memcpy and memset records
   on CUDA, the profiler SDK's memory-copy tracing on ROCm.
2. **Read NVTX and ROCTX ranges.** Done (M7, 2026-10-20): CUPTI's NVTX callbacks on CUDA, the
   profiler SDK's marker callbacks on ROCm.
3. **Collective metadata.** Done (M8, 2026-10-21): the operation, size and group of each
   collective, joined to its kernels, which turns the transfer estimate into per-operation
   bandwidth. On vLLM most collectives are its own kernels, not NCCL's or RCCL's (M7), so they
   come from the framework adapter. ruppell's NCCL profiler plugin gives NCCL's and RCCL's own
   record of their calls; RCCL needed no SDK tracing. Counts agree with nsys's NCCL ranges and
   kernels (B200, separate run).
4. **Counters on sampled steps.** The ROCm profiler SDK's counter service and CUPTI's range
   profiler, restricted to steps `RUPPELL_DETAIL` already samples, so timing steps stay clean.
5. **Hardware timestamps on Blackwell,** once a CUPTI keeps graph-replayed records with them
   (they lost them on CUPTI 130001).
6. **Merge the viewer branch,** and a columnar export (Parquet) for analysis outside Python.
7. **Leave to nsys and rocprofv3/rocprof-sys:**
   - CPU sampling, OS runtime and thread states;
   - network and storage metrics;
   - library API traces.

   Run them on a separate run when a question needs them.

## Sources

- Nsight Systems User Guide, 2026.5: <https://docs.nvidia.com/nsight-systems/UserGuide/index.html>.
  Verified there for this page:
  - hardware trace by default from Blackwell on, and `-t cuda-sw`;
  - the `--cuda-graph-trace` values and defaults, including `nvtx-live` and `nvtx-precapture`;
  - the `--pytorch` values;
  - `--cuda-flush-interval`.
- Post-Collection Analysis Guide (stats reports, analyze rules, recipes):
  <https://docs.nvidia.com/nsight-systems/AnalysisGuide/index.html>
- Installation Guide (platforms): <https://docs.nvidia.com/nsight-systems/InstallationGuide/index.html>
- Release Notes: <https://docs.nvidia.com/nsight-systems/ReleaseNotes/index.html>
- **ruppell's own records:**
  - [`feasibility-cuda.md`](feasibility-cuda.md): CUPTI, the cost per kernel, the one-subscriber
    limit, and detail sampling;
  - [`limitations.md`](limitations.md);
  - [`milestones/accuracy-study.md`](milestones/accuracy-study.md): nsys losing the end of vLLM
    runs, and hardware vs software timestamps.
