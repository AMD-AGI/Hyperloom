# M2 plan: ruppell on vLLM serving DeepSeek V4.1 Flash

> Builds the vLLM adapter and the serving schema, makes the runtime handle
> vLLM's worker processes, and shows on a short seed-42 AgentX replay of
> DeepSeek V4.1 Flash that every rank's kernels carry a scope and a GPU-defined
> step, that every step carries its load, and that `rocprofv3` in the same
> processes sees the same dispatches. Covers v0 plan items 4.2, 4.3 and 4.4
> ([`plan-v0.md`](plan-v0.md)); M3 then runs 4.5–4.7 (validation at scale,
> overhead, correctness gates). What M1 left open: [`../TODOs.md`](../TODOs.md).

Status: **draft**, 2026-09-29

## 1. Objective

At the end of M2:

- A vLLM server started under `ruppell run` writes one run directory per
  process that touches a GPU, and `ruppell export` merges them: one Chrome
  trace with one GPU process per rank, and one census.
- The vLLM adapter, loaded as a vLLM plugin, names every graph vLLM captures
  (full graphs and each segment of a breakable capture) by its batch
  descriptor, marks every engine step on the GPU timeline of every rank, and
  records each step's serving fields.
- Kernels are scoped by module path, including inside graph replays; if module
  hooks cannot see into a region, the adapter gets scopes another way or the
  region is reported as unscoped, never guessed.
- The serving schema is loaded, filled for every step, and enforced at
  export.
- The census gives GPU time per (layer, op) for every step of the run,
  grouped by the schema's fields, with the client's phase clock aligned.

## 2. What the pinned vLLM looks like (`8f400a0ac`)

Read from the tree on 2026-09-29; M2.0 confirms each point at run time.

| What ruppell needs | Where it happens in vLLM | Consequence |
| --- | --- | --- |
| Graph captures | `GPUModelRunner.load_model` (`v1/worker/gpu_model_runner.py:5313`) wraps the model in `CUDAGraphWrapper` (`compilation/cuda_graph.py`, captures with `torch.cuda.graph` per `batch_descriptor`, line ~315) for FULL graphs, and, when `VLLM_USE_BREAKABLE_CUDAGRAPH=1` (the runner's server environment sets it), in `BreakableCUDAGraphWrapper` (`compilation/breakable_cudagraph.py`) for PIECEWISE dispatch, also around the MTP drafter | Two capture paths. A breakable capture is **many graphs per descriptor**: segments (`_begin_segment`, `_end_segment`), with eager functions between them (`add_eager`) that re-run at replay. Each segment needs its own key, `descriptor/segN`; the eager parts are eager kernels and need scopes |
| Engine steps | the engine core's `step()` (`v1/engine/core.py:589`) runs in its own process; the GPU work runs in each worker's `GPUModelRunner.execute_model` (line 4188) and `sample_tokens` (line 4567) | Step markers go in the worker, around `execute_model` and after `sample_tokens`, on the stream the model runs on; one step id per scheduler step, the same on every rank |
| Per-step load | `execute_model(scheduler_output)`: `num_scheduled_tokens`, `total_num_scheduled_tokens`, `scheduled_spec_decode_tokens`, new and cached requests (`v1/core/sched/output.py:229`); the worker's `input_batch.num_computed_tokens_cpu` (`gpu_input_batch.py:170`) | The serving fields come from here, in the worker |
| Attaching without edits | `vllm.general_plugins` entry points, loaded in every vLLM process (`plugins/__init__.py:77`) | The adapter can be a plugin that wraps these methods, with no change to the pinned tree |
| Compiled regions | `compilation_config.mode`; with `STOCK_TORCH_COMPILE` the model is `model.compile(fullgraph=True)` | Forward hooks may not run inside compiled code; M2.0 checks which mode the runner's launch uses and whether hooks fire |
| Existing SDK clients | `env_override.py:97`: vLLM puts `libtorch_cpu.so` in the global scope before `import torch` so Kineto's `rocprofiler_configure` registers with the ROCm profiler SDK; `compilation/caching.py:138`: compiled Triton modules load one at a time on ROCm because the SDK's `dl_iterate_phdr` deadlocks concurrent loads | ruppell shares every worker with a second SDK client (Kineto) and must not break it or be broken by it |

## 3. Fixed baseline

- vLLM `8f400a0ac` (pure v2+v6), the image by digest (on Cluster A:
  `/data/dsv41/image/vllm-openai-rocm-960228cf.tar.zst`), the
  runner's launch flags and `server_env_blind.list`, DeepSeek V4.1 Flash
  weights (`/data/models/DeepSeek-V4.1-Flash`), TP4, speculative
  decoding with 5 tokens and synthetic acceptance, as the runner pins them.
- `ruppell` built **inside the container**, against the container's ROCm and
  profiler SDK; the host's ROCm 7.0.2 build from M1 is not used there.
- AgentX, seed 42, C8, `NUM_PROMPTS` 320, a shortened `DURATION` (M2 is
  functional; timing claims are M3's).
- Cluster: Cluster A (MI350). M2 makes no performance claim, and no number from it
  is compared with MI355X results (runner AGENTS.md §11).
- What may change: `ruppell` only. The vLLM tree is not edited; if the plugin
  route fails, M2 stops for a decision (§6).

## 4. Work, in order

### M2.0 Facts first (no ruppell changes)

In the container on a node, record in `docs/feasibility-m2.md`:

- the container's ROCm and profiler SDK versions; `ruppell` builds and the M1
  probe passes inside the container (tests (a)–(e));
- the process layout of the pinned server at TP4: which processes initialize
  HIP (API server, engine core, workers), their pids and ranks;
- the served configuration from the boot log: compilation mode, cudagraph
  mode, breakable graphs on or off, capture sizes, graph and segment counts;
- whether module forward hooks fire during capture in that configuration
  (a one-module test inside the server's model, or a smaller model with the
  same flags);
- whether ruppell's tool loads next to Kineto's client in a worker
  (`[ruppell] tracing` in each worker's log, records written), with and
  without vLLM's torch-symbol promotion;
- whether SDK timestamps from two worker processes on one node share a
  timebase (clock pairs from both, one wall-clock read apart);
- the tool's per-dispatch cost on a graph-replaying workload (the M1 probe at
  two lengths inside the container, and a graph-heavy variant), for §9.4;
- which streams a served step uses besides the compute stream, and whether
  `execute_model` is called the same number of times on every rank, for §9.1
  and §9.2;
- that the drafter has its own wrapper instance and runs inside
  `sample_tokens`, for §9.3.

### M2.1 Runtime: one run directory per process

- The tool writes to `$RUPPELL_OUT/<host>-<pid>/`; `run.json` stays at the
  top. A process that never touches HIP writes nothing.
- `ruppell.context(rank=..., role=...)` names the process; the adapter calls it.
- Export merges all process directories: one Chrome process per GPU rank
  (named by rank), host tracks per process, one validation block per rank and
  one overall.
- Cross-process alignment: SDK timestamps are used as they are when M2.0
  shows a shared timebase; otherwise through clock pairs; export refuses a
  merge whose clock pairs disagree beyond a stated bound.
- Step merge across ranks (§9.1): export pairs ranks by step id and checks
  each step's fingerprint on every rank; a mismatch refuses the merge and
  names the first divergent step.
- Buffer pressure (§9.4): the tool records time spent waiting on a full
  buffer and records per second, in `meta.json` and the validation block.
- Offline test: two synthetic process directories, merged; one with a
  diverging fingerprint, refused.

### M2.2 The vLLM adapter

A package `ruppell.adapters.vllm`, registered as a `vllm.general_plugins`
entry point, active only when `RUPPELL_VLLM=1` and the runtime is loaded (a
worker without the runtime refuses to start with the flag set). In each
worker it wraps, without editing vLLM:

- `GPUModelRunner.load_model`: after the model loads, `ruppell.instrument()`
  on the target model and, with the prefix `drafter.`, on the drafter (§9.3);
  the adapter notes which graph wrapper instance belongs to which role; then
  `ruppell.context(rank, tp, model, cudagraph_mode, ...)`;
- `CUDAGraphWrapper` capture: `ruppell.capture(key)`, with the key made of
  role, runtime mode and batch descriptor (§9.3);
- `BreakableCUDAGraphCapture._begin_segment`: a capture key per segment
  (role, mode, descriptor, segment number), and a scope around each
  `add_eager` function;
- `GPUModelRunner.execute_model` and `sample_tokens`: `step_begin` with the
  serving fields before the forward, `step_end` after sampling is enqueued,
  on the compute stream. The step id is a counter of `execute_model` calls in
  each worker, and a fingerprint of the scheduled work (request ids and token
  counts, hashed) goes in the step's fields for the merge check (§9.1). The
  CPU window of each step (from `execute_model` entry to `sample_tokens`
  exit) is recorded so side-stream work can be charged to the step that
  launched it (§9.2). Startup, weight load and capture run inside named steps
  (`startup`, `capture`) so every kernel lands in a step;
- **scopes in compiled regions**, if M2.0 finds hooks do not fire there:
  first, module paths from the compiled graph's node metadata
  (`nn_module_stack`) at compile time, attached to the launches recorded while
  capturing; second, PyTorch `RecordFunction` callbacks during capture. The
  adapter reports which source each scope came from.

Every wrapped method checks its target's signature at load; a mismatch
disables the adapter for that process with a message saying which hook and
why, rather than tracing half the points.

### M2.3 The serving schema

- A loader and validator for `schemas/serving.yaml` (no new dependency: the
  file is kept to the subset `json` can read after a trivial conversion, or
  the schema moves to JSON).
- Fields per step, computed by the adapter from `scheduler_output` and the
  input batch: running requests, rows (scheduled tokens, speculative rows
  included), context length per request, kind (prefill, decode, mixed), graph
  key or eager, cudagraph mode; run fields: model, TP, rank, the client's
  phase clock when given.
- Export refuses a run in which a step lacks a required field, and names the
  steps.

### M2.4 The census

`python -m ruppell census RUN`: per step and per rank, GPU time by (scope,
op), busy time (union of kernel intervals) and the step period kept apart,
the schema fields, and the aiperf phase (warmup, profiling) the step falls in.
Kernels on the compute stream are placed by GPU time; kernels on side streams
by the step that launched them, with a count of those whose GPU time falls
outside their owning step (§9.2). Target and drafter are split by scope
prefix (§9.3). Output as JSON Lines (no new dependency), one line per step
and rank.

### M2.5 Export at serving size

A C8 AgentX replay runs tens of thousands of steps with thousands of kernels
each; one Chrome JSON of all of it is too large to open. Export gains a window
(`--steps A:B` or `--time A:B` on the phase clock) for the trace, while the
census always covers every step.

### M2.6 A small model first

vLLM from the same image serving a small public model (for example
Qwen2.5-0.5B), with the same cudagraph and breakable settings where they
apply, at TP1 then TP2, under a simple fixed prompt set: the adapter loads,
every process writes its own directory, captures are keyed, steps are marked
on every rank, scopes resolve, the schema validates, the witness matches.
This is where most adapter iterations happen: minutes per server start
instead of the tens of minutes DSV4.1 takes.

### M2.7 DeepSeek V4.1 Flash, short AgentX replay

The pinned server at TP4 on one MI350 node, under `ruppell run` with
`RUPPELL_VLLM=1`, a discarded warmup replay, then one short C8 replay at
seed 42; export a window of the profiling phase, run the census, run the
witness (in-process `rocprofv3`) on each rank.

### M2.8 Correctness spot check

Greedy output of a fixed prompt set identical with `RUPPELL_VLLM=0` and `=1`
(M3 runs the full correctness gate).

## 5. Exit criteria

On the M2.7 run (and the M2.6 runs, which must pass first):

- **Processes**: one run directory per GPU process; every rank present in the
  merged trace; the merge's clock check passes.
- **Graphs**: every graph vLLM captured (full and every breakable segment) is
  keyed; every replay is joined to its capture or refused with a reason; no
  graph silently unjoined.
- **Scopes**: every kernel's attribution is reported by source (graph
  registry, eager scope, compile-time metadata, none); decode-step GPU time
  attributed to a scope is at least 95% on every rank, or the unattributed
  kernels are named with the reason (this is the plan's coverage gate, which
  M3 enforces at C8 and C32).
- **Steps**: every compute-stream kernel after the server is up lies in
  exactly one step on its rank; every side-stream kernel has an owning step,
  and the number running outside it is reported; startup and capture work
  lies in named steps; step ids agree across ranks and every step's
  fingerprint matches on every rank (§9.1).
- **Keys**: no two captures share a key; target and drafter captures are
  distinguished by role (§9.3).
- **Volume and pressure**: record volume per rank, records per second, and
  time spent waiting on a full buffer are reported for the M2.7 run (§9.4).
- **Schema**: every step on every rank carries every required serving field;
  the time-weighted mean of running requests over the profiling phase agrees
  with the in-flight count computed from aiperf's own export within the
  replay's spread.
- **Witness**: on every rank, `rocprofv3` in the same process sees the same
  dispatches (counts equal, identical timestamps).
- **Viewers**: the windowed trace opens in Perfetto's trace processor and
  `chrome://tracing`'s importer without errors or warnings.
- **Correctness**: M2.8 identical.

## 6. Stop conditions

- The SDK or the tool cannot be loaded in vLLM's worker processes, or it
  breaks Kineto's client or the server: stop and report what fails.
- The capture or step points cannot be reached by a plugin: stop; the
  alternative (one small hook per point on a branch of the pinned vLLM,
  behind the flag) needs the operator's decision, because it changes the
  measured tree.
- Scopes cannot be recovered for compiled regions by either route: report
  attribution by source, and stop for a design decision before M3.
- A node is lost mid-job: keep what is written, resubmit; no result is taken
  from a partial job.

## 7. Artifacts

```text
ruppell: docs/feasibility-m2.md, docs/milestones/m2.md
Cluster A:    ~/ruppell-work/runs/<job>/   per-job results (reports, census, windowed traces, logs)
```

Raw record files stay on node-local disk; summaries, the census and windowed
traces come back to `/home`. Every result names the `ruppell` commit and the
runtime library digest built in the container.

## 8. Execution setup (Cluster A)

- **Jobs, never held nodes**: every step is an `sbatch` job that does its work
  and exits, at most 4 hours including provisioning; submit the next job while
  one runs, never more than one ahead (runner `docs/clusters/cluster-a.md`).
- **Docker**: the pinned image needs a node with a working Docker daemon (one
  M1 node, `c12-08`, had none); M2.0's first job finds one and records it.
  Containers are removed on exit (`trap 'docker rm -f …' EXIT`).
- **Weights**: staging DeepSeek V4.1 Flash to node-local disk takes about 25
  minutes; nodes that already hold it (`c16-08`, `c23-08`, `e07-18`,
  `e12-08` as of 2026-09-25) save that, if idle. The runner's
  `knob-ab/provision_ab15.sh` provisions image and weights and is the
  starting point.
- **Budget per DSV4.1 job**: provision 3–25 min, server start 5–15 min,
  warmup replay about 15 min, one short C8 replay about 20 min, export and
  census on the node about 10 min: one M2.7 attempt per 4-hour job with
  margin.
- **Parsing**: records are parsed on the GPU node; only summaries leave it.

## 9. Decisions

Recorded 2026-09-29 by the operator, from the options below. Each decision
names what would show it wrong; M2.0 and M2.6 check them first. Facts behind
them, from the pinned vLLM: the scheduler output carries no step counter;
with async scheduling the model runner copies outputs on side streams
(`async_output_copy_stream`, draft-token copy streams); the MTP drafter runs
inside `sample_tokens` (`propose_draft_token_ids`) and is wrapped by its own
`BreakableCUDAGraphWrapper`, so its captures use the same kind of
batch-descriptor keys as the target model's.

### 9.1 Step id: a per-worker counter, checked by a fingerprint

The step id must be identical on every rank so that step N lines up across
the four GPUs.

| Option | For | Against |
| --- | --- | --- |
| A. The engine's counter, attached to the scheduler output in the engine process | one authoritative number | the scheduler output is serialized to the workers; an added attribute may not survive, and making it survive touches vLLM's message types |
| **B. Each worker counts `execute_model` calls** (chosen) | no cross-process plumbing | silent drift if one rank skips or repeats a call, after which every later step is mis-paired |
| **C. A fingerprint of what was scheduled** (chosen, as the check on B) | identical on every rank, since every rank receives the same scheduler output; pairs ranks by content | a small hash per step in Python; two identical consecutive steps would collide (rare: computed-token counts change every step) |
| D. Time order only | nothing to build | drift as in B, and undetectable |

**Decision: B, checked by C.** The per-worker counter is the step id. A
fingerprint (request ids and scheduled token counts, hashed) goes in each
step's fields. Export checks that every rank has the same fingerprint for
every step id, and refuses the merge where they differ, naming the first
divergent step. No vLLM change. *Shown wrong if* ranks legitimately run
different numbers of `execute_model` calls (then step ids must come from the
fingerprint alone).

### 9.2 Steps under async scheduling: GPU time on the compute stream, CPU ownership on side streams

Markers enqueued on the model's compute stream execute in enqueue order, so
steps stay ordered there even when the CPU prepares step N+1 while the GPU
runs step N. The risk is side-stream work: an output copy belonging to step N
can run during step N+1.

| Option | For | Against |
| --- | --- | --- |
| **A. GPU time relative to the compute-stream markers** (chosen, for the compute stream) | simple; M1's rule | side-stream work can land in the wrong step or straddle a boundary |
| **B. CPU ownership**: a dispatch belongs to the step whose `execute_model`…`sample_tokens` window on the CPU launched it (via the launching call's timestamp) (chosen, for side streams) | exact ownership whatever stream ran it | two definitions to explain; work launched late in step N and run in N+1 is charged to N (right for ownership, not for timing) |
| C. Markers on every stream the step uses | exact per-stream boundaries on the GPU | more marker kernels; the adapter must know every stream, and a missed one is silently wrong |

**Decision: A for the compute stream, B for side streams, and report where
they disagree.** Steps stay defined by GPU time on the compute stream; a
kernel on any other stream gets the step that launched it, and the census
and validation block count the kernels whose GPU time falls outside their
owning step. *Shown wrong if* that count is large enough to matter for the
census, in which case C is reconsidered.

### 9.3 The drafter: its role in every capture key, and its own scope prefix

| Option | For | Against |
| --- | --- | --- |
| **A. Capture key = role (target or drafter), runtime mode, batch descriptor, segment** (chosen) | unambiguous keys; each replay joins to the right capture | the adapter must identify the wrapper instances at load (it wraps both, so it can) |
| B. Sub-steps: markers around the drafter inside each step | the draft phase separated on the timeline | two more marker kernels per step; "step" no longer means one scheduler step |
| **C. Scopes only: instrument the drafter with the prefix `drafter.`** (chosen) | cheap; the census splits target and draft by scope | no timeline boundary between them |

**Decision: A and C.** A is required: without the role, target and drafter
keys collide and joins break. C gives the target/draft split in the census.
B is left until someone needs the draft phase as a timeline boundary.
*Shown wrong if* the drafter shares a wrapper instance with the target, or
runs outside `sample_tokens`.

### 9.4 Record volume and overhead: record everything in M2, and measure

Estimate: runner Plan 19 counted about 2,800 kernel launches per decode step.
A C8 replay of about 900 s at about 28 steps per second is about 25,000 steps,
or about 70 million dispatch records: about 7 GB per GPU at 96 bytes each,
about 27 GB for four GPUs, written at about 8 MB/s per GPU. The larger risk is
cost: at M1's 2.5 µs per dispatch, 2,800 dispatches would be about 7 ms of
tool time per step, against a step of about 12 ms, if that cost lands on
graph-replayed kernels at all.

| Option | For | Against |
| --- | --- | --- |
| **A. Record everything, as built; window the trace, census every step** (chosen for M2) | complete; nothing new to build | about 7 GB per GPU per replay; parse on the node; overhead unknown |
| B. Smaller records (drop redundant fields, delta-encode timestamps; 32–48 bytes) | about half the volume | a format change the reader must follow |
| C. Aggregate in the tool: per-step sums per kernel, full detail only for chosen windows | orders of magnitude less data | the graph join and scope mapping must move online, into C++; the hardest option |
| D. Sample steps: dispatch tracing on for 1 step in N (start and stop the SDK context) | cuts volume and overhead | no longer a census of every step; context start/stop cost and correctness unknown |

**Decision: A for M2, with measurement built in; B, C or D are chosen in M3
from the numbers.** Two additions: the tool counts time spent waiting on a
full buffer and records per second (M2.1), and M2.0 measures per-dispatch
overhead on a graph-replaying workload. *Shown wrong if* the M2.7 run cannot
complete, or the buffer blocks the server measurably; then D (or C) is
pulled into M2.
