# ruppell

See executions through a Rüppell's vulture's eyes.

ruppell is a GPU tracer for programs that run their hot path as captured graphs, such as
LLM serving. It attributes every kernel, eager or replayed from a graph, to the code scope
that launched it (module path, op, shape). It assigns each kernel to a step defined on the
GPU timeline, and checks its own numbers against an independent collector.

The PyTorch profiler misleads on these workloads. Inside a graph replay the call sites and
shapes are gone, and CPU step ranges do not contain the GPU work they launched. The
measurements: [`docs/background.md`](docs/background.md).

Repository: [`the ruppell repository`](ruppell's repository (AMD internal))
(moved from `<fork>/ruppell` on 2026-10-19 with its history; commit ids are unchanged).

## Platforms

| | ROCm | CUDA |
| --- | --- | --- |
| Records through | ROCm profiler SDK | CUPTI (API 130001) |
| Tested on | MI350X, MI355X; ROCm 7.0.2 to 7.2.3 | B200; CUDA 13.0 |
| Independent check | `rocprofv3`, in the same process | Nsight Systems, on a separate run |

Workloads traced so far:
- plain PyTorch with captured graphs;
- vLLM serving through its v2 model runner (DeepSeek V4.1 Flash at TP4, Qwen3);
- DLRMv4 training on TorchRec.

What was verified, per milestone: [`docs/status.md`](docs/status.md). Every validation run's numbers on
one page: [`docs/validation.md`](docs/validation.md).

## Architecture

```text
ruppell run --out RUN -- <command>
  loads libruppell_tool.so into every process of <command>
  (ROCP_TOOL_LIBRARIES on ROCm, CUDA_INJECTION64_PATH on CUDA)

in each traced process:
  your program ── PyTorch / vLLM ── HIP or CUDA runtime ── GPU
      │                                   │
      │                      ROCm profiler SDK or CUPTI
      │                                   │  dispatches, API calls, graph capture
      │                                   │  and launch, kernel names, clock pairs
      │                                   ▼
      ├── ctypes: scopes, graph keys ──► libruppell_tool.so ──► RUN/<host>-<pid>/
      └── ctypes: step markers ────────► libruppell_marker.so (a one-thread marker kernel)

afterwards, anywhere (numpy):
  ruppell export | census | birdeye | tracelens | witness   RUN
```

- **The tool library** (`runtime/src/tool.cpp`, `tool_cuda.cpp`) records, through a lossless
  buffer:
  - every kernel dispatch and runtime API call;
  - the graph lifecycle: capture, instantiate, launch, and every launch made while capturing;
  - kernel names.
  It writes compact binary records and serializes nothing in the traced process. Details:
  [`runtime/README.md`](runtime/README.md).
- **The marker library** (`runtime/src/marker.hip`, `marker.cu`) holds the step marker kernel.
- **The Python API** (`python/ruppell/api.py`) is how a program or a framework adapter says
  what is running: it pushes scopes, names graph captures and launches step markers.
- **The analysis** (`python/ruppell/engine.py`) joins the records of every process and checks
  them across ranks. Its outputs:
  - the Chrome trace;
  - the census;
  - birdeye's reports;
  - the TraceLens export;
  - the witness comparison.

### How a kernel gets its scope and its step

| Kernel | How it is attributed |
| --- | --- |
| Launched eagerly | The scope active on the launching thread is pushed as the external correlation id, which the collector stamps on the dispatch. |
| Replayed from a graph | Capture runs the program's Python, so every launch made while capturing is recorded with its scope. A replay's dispatches are joined to those launches by graph node on CUDA, and by position on ROCm. On ROCm a graph is joined only if every replay runs the capture's kernel sequence; otherwise it is refused, with the reason. |
| Any | A step is the GPU time between two marker kernels (`step_begin`, `step_end`) on the program's stream. Kernels are assigned by GPU timestamps, so a CPU running ahead does not matter. |

### Three layers

| Layer | Where | Written once per |
| --- | --- | --- |
| Runtime | `runtime/`, `python/ruppell/` | platform (ROCm, CUDA) |
| Framework adapter | `python/ruppell/torch_hooks.py`, `python/ruppell/adapters/<framework>/` | framework (plain PyTorch, vLLM, DLRM) |
| Workload schema | [`schemas/`](schemas/README.md) | workload type (serving, training) |

ruppell builds on the platform's documented tracing interface and does not reimplement how
timestamps come off the GPU. Design of record: [`docs/design.md`](docs/design.md).

### A run directory

| File | Written by | Holds |
| --- | --- | --- |
| `dispatch.bin` | tool | one record per kernel dispatch |
| `copies.bin` | tool | one record per memory copy (and set, on CUDA) |
| `ranges.bin`, `ranges.tsv` | tool, with `RUPPELL_NVTX=1` | one record per NVTX (ROCTX) range or mark, when it closes; range names by domain |
| `nccl.tsv` | ruppell's NCCL profiler plugin, which `ruppell run` loads (`RUPPELL_NCCL=0` turns it off) | NCCL's or RCCL's own record of each call: communicators, API calls, launches, and tasks with their count, datatype, algorithm, protocol, channels and channel span on the GPU |
| `api.bin`, `api_ops.tsv` | tool | one record per runtime API call; operation names |
| `capture.bin` | tool | graph capture, instantiate, launch; launches made while capturing |
| `kernels.tsv` | tool | kernel id to symbol |
| `defs.tsv`, `steps.tsv` | tool, from the Python API | scope, graph-key and collective-call definitions (the vLLM adapter's; `RUPPELL_COLLECTIVES=0` turns them off); per-step fields |
| `clock.tsv`, `self.bin`, `meta.json` | tool | clock pairs; the tool's own time; counts and drops (rewritten every 5 s) |
| `run.json` | `ruppell run` | command, commit, library digests |
| `trace.json`, `report.json` | `ruppell export` | Chrome trace; validation block |
| `witness.json` | `ruppell witness` | the comparison with the independent collector |
| `census.jsonl`, `census_summary.json`, `census_scopes.*` | `ruppell census` | per step and rank: period, busy time, GPU time by op and scope |
| `birdeye/` | `ruppell birdeye` | `REPORT.md`, one CSV per table |

Each process writes its own `RUN/<host>-<pid>/`, and the outputs sit at the top.

## Usage

### Build

```bash
bash runtime/build.sh            # libruppell_tool.so, libruppell_marker.so, graph_probe
export PYTHONPATH=$PWD/python    # or: pip install -e .  (the Python side only)
```

On CUDA: `CUDA_PATH=/usr/local/cuda-13.0 bash runtime/build.sh`, which builds the CUPTI tool.

### Trace a program

```bash
python -m ruppell run --out RUN -- python my_program.py     # records to RUN/
python -m ruppell export RUN -o RUN/trace.json              # Chrome trace and validation block
```

Open `trace.json` in [ui.perfetto.dev](https://ui.perfetto.dev) or `chrome://tracing`.

### Instrument a PyTorch program

```python
import ruppell
ruppell.init()                                  # refuses unless started by `ruppell run`
ruppell.instrument(model, prefix="model")       # scope every module by its path
with ruppell.scope("sampler", op="topk"):       # any code outside modules
    ...

ruppell.step_begin(0, phase="capture")          # markers on the current torch stream
with ruppell.capture("decode.b16", batch=16):   # name the next graph captured on this thread
    with torch.cuda.graph(g):
        model(x)
ruppell.step_end(0)

for step in range(1, n + 1):
    ruppell.step_begin(step, phase="decode", running=4)   # fields go to the step's args
    g.replay()
    ruppell.step_end(step)
```

Scopes nest, and the innermost wins. A complete example is
[`examples/torch_graph_demo.py`](examples/torch_graph_demo.py). Without Python, a program
calls the same C API through `dlsym`, as
[`tests/probes/graph_probe.hip`](tests/probes/graph_probe.hip) does.

### vLLM

```bash
python -m ruppell run --out RUN -- env RUPPELL_VLLM=1 vllm serve ... --shutdown-timeout 60
python -m ruppell census RUN --phase profiling=START:END              # every step; realtime s
python -m ruppell export RUN --steps 12590:12610 -o RUN/trace.json    # a window of engine steps
```

- **How it works:** the plugin (`python/ruppell/adapters/vllm/`) wraps the v2 model runner, its
  graph captures and every module, and records each engine step's scheduler state. vLLM
  itself is unchanged.
- **Requirements:**
  - vLLM's v2 model runner with breakable graphs:
    `VLLM_USE_V2_MODEL_RUNNER=1 VLLM_USE_BREAKABLE_CUDAGRAPH=1`, as
    [`tests/cuda/serve_vllm.sh`](tests/cuda/serve_vllm.sh) sets them;
  - `--shutdown-timeout 60`. At vLLM's default of 0, the workers are killed before their last
    flush.
- **Background:** why vLLM needs an adapter is in
  [`adapters/vllm/README.md`](adapters/vllm/README.md); adapting another framework is in
  [`docs/adapting-frameworks.md`](docs/adapting-frameworks.md).

### Check against an independent collector

```bash
# ROCm: both tools in one process, compared dispatch by dispatch
rocprofv3 --kernel-trace --output-format csv -d W -o %pid% -- \
    python -m ruppell run --out RUN -- python my_program.py
python -m ruppell witness RUN W

# CUDA: CUPTI takes one subscriber, so nsys traces a separate run with markers only;
# a second nsys run bounds the comparison by nsys's own run-to-run spread
nsys profile -t cuda --cuda-graph-trace=node --export sqlite -o W/nsys \
    python -m ruppell run --witness --out W -- python my_program.py
python -m ruppell witness RUN W --spread W2
```

A served model under `rocprofv3` must be stopped process by process, or the GPU processes end
without writing their files: [`docs/limitations.md`](docs/limitations.md).

### Read the results

```bash
python -m ruppell birdeye RUN                                     # RUN/birdeye/REPORT.md and CSVs
python -m ruppell birdeye-compare CMP RUN_A/birdeye RUN_B/birdeye --tags a,b
python -m ruppell tracelens RUN --steps 12452:12472 -o RUN/tracelens   # for AMD TraceLens
```

- **birdeye** is a built-in micro TraceLens. It reads a ruppell run, a PyTorch profiler trace,
  an nsys SQLite export or a `rocprofv3` CSV, and writes TraceLens's tables plus ruppell's
  step table: [`docs/birdeye.md`](docs/birdeye.md).
- **The TraceLens export** puts every kernel, replayed or eager, under a `cpu_op` named by its
  scope: [`docs/tracelens.md`](docs/tracelens.md).
- **Profiling a server:** follow the
  [`ruppell-profile` skill](.claude/skills/ruppell-profile/SKILL.md).

## Limitations

The main ones:
- **Full tracing slows the host** by about 2 µs per dispatch, which moves a server's operating
  point. For serving, sample with `RUPPELL_DETAIL`. Never compare a traced run's latency with
  an untraced one.
- **Collective time includes waiting for other ranks,** and tracing changes which rank waits.
  Use the transfer estimate, not per-rank wait.
- **The vLLM adapter covers the v2 model runner,** with breakable graphs or with vLLM's
  compiled path (`-O2`, since 0.9.0: compiled pieces, Inductor's launch sites and the split
  ops get their modules' scopes). Hooks do not fire inside other `torch.compile` regions.
- **Scopes are modules, not aten ops,** unless the CPU op tree is on
  ([`docs/cpu-ops.md`](docs/cpu-ops.md)). Without it there are no FLOP rates.
- **ruppell records timings only,** with no hardware counters.

Each output carries the limitations that apply to its run
([`python/ruppell/limits.py`](python/ruppell/limits.py)). The full list, with evidence:
[`docs/limitations.md`](docs/limitations.md). Open work: [`TODOs.md`](TODOs.md).

## Layout

```text
runtime/          tool libraries (ROCm, CUDA), marker kernels, record layouts, build.sh
python/ruppell/   API, CLI, PyTorch hooks, analysis engine, export, census, witness, birdeye,
                  adapters (vllm, dlrm)
schemas/          workload schemas (serving, training)
examples/         torch_graph_demo.py
tests/            offline tests, GPU probes, cluster jobs per milestone and plan
docs/             design, status, limitations, milestones, analyses (index: docs/README.md)
```

Rules for working on this repository: [`AGENTS.md`](AGENTS.md).
