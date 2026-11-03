# Feasibility (roadmap M1.1, plan 4.0)

Answers from runs on AMD Instinct MI350 (gfx950) nodes of the Cluster A cluster,
2026-09-29, ROCm 7.0.2.2, ROCm profiler SDK 1.0.1. Each answer names the job
that showed it; logs are under `~/ruppell-work/runs/<job>/` on Cluster A.

| Question | Answer | Evidence |
| --- | --- | --- |
| SDK present on the host | Yes: headers, CMake package, `librocprofiler-sdk.so.1.0.1`, `rocprofv3` | probe job 47829 |
| A tool of ours loads | Yes, through `ROCP_TOOL_LIBRARIES` (the SDK calls `rocprofiler_configure`) | 47868 onward: `[ruppell] tracing to …` |
| Dispatch records for directly launched kernels | Yes, with GPU start and end, correlation to the launching call, dispatch parameters | 47882 |
| Dispatch records for kernels launched by `hipGraphLaunch` | Yes. Their correlation id is the `hipGraphLaunch` call's. **No graph or node id** in the record (`rocprofiler_kernel_dispatch_info_t` has none) | 47882: 8000 of 8000 replayed dispatches correlated to their launch |
| Capture visible through HIP API callbacks | Yes: `hipStreamBeginCapture`/`EndCapture` (with the graph handle), `hipGraphInstantiate*` (exec to graph), `hipGraphLaunch`, and every kernel launch made while capturing | 47882 onward: 8 of 8 captured launches (probe), 10 of 10 (PyTorch demo) |
| Replay sequence stable | Yes: 1000 of 1000 replays ran the same kernel sequence (probe); 200 of 200 (demo) | 47882 onward |
| **Join mode** | **Position in the replay, ordered by dispatch id**, after checking that every replay has as many dispatches as the capture has launches and the same kernel sequence; otherwise the graph is refused with the reason | recorded in `validate/core.py` |
| Marker kernel visible, in order, with its code | Yes: every marker dispatch carries the external correlation value pushed around its launch | 47888 onward: 2002 of 2002 markers decoded |
| Scopes on eager launches | Yes, through external correlation ids pushed per thread | 2002 of 2002 (probe), 246 of 246 (demo) |
| Per-dispatch cost | About 2.2–2.5 µs per dispatch, plus about 250 ms fixed per process (SDK and tool start, flush, exit), measured with two probe lengths | 47936, 47944 (`overhead.txt`) |
| `ruppell` and `rocprofv3` in one process | Yes, when `ruppell run` joins `ROCP_TOOL_LIBRARIES` and also preloads the tool (rocprofv3 initializes the SDK in a library constructor). Both then see identical dispatches: same ids, same kernel ids, same timestamps | 47916 onward |
| Served cudagraph configuration | Not applicable to M1 (no vLLM yet); M2 records it | — |

## Surprises worth keeping

- **The SDK finalizes after the tool's static destructors.** Any non-trivial
  global in the tool (a `std::thread`, a `std::string`) must be leaked, or the
  finalizer runs against destroyed objects (job 47868: `std::terminate`,
  truncated files).
- **Kernels on one queue overlap in time** by up to a few hundred ns: the next
  dispatch starts before the previous completion is recorded. The Chrome
  export puts them on overlap lanes rather than adjusting timestamps.
- **A replay's first kernels start before `hipGraphLaunch` returns.** Flow
  arrows must start at the beginning of the launching call.
- **Kernel names from code objects are descriptor symbols** (`….kd`), and
  demanglers differ in style; compare kernels by the SDK's kernel id where
  possible.
- **Some Cluster A nodes have no writable `/scratch`**; jobs pick the first
  writable of `/scratch/$USER`, `/var/tmp/$USER`, `/tmp/$USER`.
- **PyTorch for this host**: AMD's lightweight builds for ROCm 7.0.2
  (`torch-2.8.0+rocm7.0.2.lw`, Python 3.9, from repo.radeon.com with the
  repository as find-links for Triton) link the system ROCm, so the traced
  process has one HIP runtime (`/opt/rocm-7.0.2.2/lib/libamdhip64.so.7`).
