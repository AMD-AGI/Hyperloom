# DeepSeek V4.1 Flash Graph-Aware Tracer Plan (ruppell v0)

> This is the plan of record. It moved here on 2026-09-29 from
> `jiaweic-runner` (where it was Plan 23; that file now points here). To run
> it on a runner mesh, submit this file; the mesh pins its own copy. The
> milestones that build toward it are in [`roadmap.md`](roadmap.md).

> Builds `ruppell` v0, our own tracer, in all three layers: a runtime layer for
> ROCm, a framework adapter for vLLM, and a workload schema for serving. It is
> validated on seed-42 AgentX against independent witnesses, and its overhead is
> measured. The root-cause analysis is in [`background.md`](background.md) and
> the design in [`design.md`](design.md); the code
> lives in [`<fork>/ruppell`](an internal fork), which moved to
> [`the ruppell repository`](ruppell's repository (AMD internal)) on 2026-10-19. It changes
> no served behaviour and makes no performance claim about the model; its
> product is an instrument with a stated coverage, agreement and overhead.
> This is an internal execution document, not published documentation.

Status: **draft**, not submitted

## 0. Authority

The gates in §5, the stop conditions in §6 and the order of §4 are fixed by the
operator (AGENTS.md §12). Choices the plan leaves open (a hook's placement, a
record layout, the scope source) are recorded with `intake decide`.
`ruppell`'s outputs are evidence; none of them becomes a gate on another plan.

## 1. Objective

`ruppell` v0 exists as one package with three working layers, and on seed-42
AgentX at C8 and at C32, over the whole measured phase of one replay each, it:

- attributes at least 95% of GPU kernel time in decode steps to a (layer, op)
  key, per step, with steps defined on the GPU timeline and each step carrying
  its load and regime;
- agrees with an independent collector on the same work;
- in always-on mode, moves paired per-request ITL by less than the node's own
  tracer-off spread;
- emits traces the Perfetto UI and TraceLens open.

## 2. Evidence

The PyTorch profiler's traces have failed this workload in measured ways (`background.md`
§2):
- call sites and shapes vanish in graph replay (Plans 11, 13);
- many kernels share one name (Plan 19);
- 0 of 10,874 indexer kernels fall inside GPU step spans, which start a median
  23.8 ms after their CPU step; at C8 the window is 39.5% GPU-busy, a mixture
  of 193 steps at about 91% and 7 that hold 91.6% of the idle (Plan 19);
- C32 windows profiled the warmup (Plans 10–15);
- profiler teardown blocked about 19 s and read as a stall (D04);
- cell windows did not compose into the served step (Plan 19, D01).

Plan 19's `mxfp8_posjoin.py` joined one kernel family to its call sites by
replay position, refusing to join unless its preconditions held: the join
works where its preconditions hold. The operator chose on 2026-09-29 to own the
runtime layer rather than build on Kineto or `rocprofv3` (`design.md` §4.1).

### 2.1 The workload this targets

From r01 and Plan 16 (AGENTS.md §8): decode is 87–95% of request time; C8
averages 1.23 running requests with 6–18 rows per step; C32 is two regimes
(5.8 running, then 21.7 after the phase's 300 s mark) with 30–160 rows. Decode
runs graph-captured; 4.0 records the cudagraph mode and graph count the pinned
launch configuration actually uses (D02 found `FULL_AND_PIECEWISE` as the vLLM
default and ran its points under `FULL_DECODE_ONLY`). Nothing measured here
depends on synthetic speculative acceptance.

### 2.2 The expected gain against the noise floor

Not a performance plan: coverage and agreement have no noise floor. The floor
that matters is for overhead: always-on `ruppell` must move paired
per-request ITL by less than the node's tracer-off spread, measured in 4.7.

## 3. Fixed baseline

- vLLM `8f400a0ac`, InferenceX `16280c78`, the image by digest, the harness and
  `server_env_blind.list` as in `sources.lock.json`; AgentX seed 42,
  `NUM_PROMPTS` 320, `DURATION` 600.
- `ruppell` on its default branch, from its first commit. Every build used for
  a result is recorded by commit and by the digest of the built library.
- The vLLM adapter reaches vLLM through a plugin if vLLM exposes the needed
  hook points without code changes; otherwise one commit per hook on a branch
  from `8f400a0ac`, behind one env flag, `VLLM_RUPPELL`, default off, declared
  in `vllm/envs.py` (AGENTS.md §7, §10). Which of the two, by `intake decide`
  after 4.0. With the flag off, the tree behaves as the base.
- What may change: `ruppell`, and the adapter's hooks. Nothing else.

## 4. Work items, in this order

### 4.0 Feasibility (no server)

In the container on the node, with standalone programs only:

- **SDK available.** The ROCm version of the image; the profiler SDK's
  headers and library present; a minimal tool library of ours loads into a
  HIP program and receives callbacks.
- **Dispatch records.** Kernels launched directly and kernels launched by
  `hipGraphLaunch` both produce dispatch records with GPU start and end
  timestamps and a correlation to the launching API call. Record the exact
  fields, including any graph or node id.
- **Capture visibility.** HIP API callbacks for `hipStreamBeginCapture`,
  `hipStreamEndCapture`, `hipGraphInstantiate`, `hipGraphLaunch`, and for the
  kernel launches made while a stream is capturing.
- **Replay stability.** A graph of at least eight distinct kernels replays the
  same sequence every time over at least 1,000 replays (the position-join
  precondition, as `mxfp8_posjoin.py`'s P1).
- **Markers.** A one-thread marker kernel enqueued on the stream appears in
  dispatch records, in order, with its step id readable.
- **Cost.** Per-dispatch overhead of the minimal tool on a tight replay loop.
- **The served configuration.** The cudagraph mode and number of captured
  graphs, from a boot log of the pinned launch configuration.

Produces `FEASIBILITY.md`. Records with `intake decide`: the join mode (node
id, or position), the adapter route (plugin or hook commits), and the scope
source (our scope API, PyTorch `RecordFunction` callbacks, or both).

### 4.1 Runtime layer (ROCm)

In `ruppell/runtime/` and `ruppell/python/`: the SDK tool library; the graph
registry (launches recorded per graph at capture, joined at replay); step
markers; per-thread ring buffers with a drop counter, drained by a background
thread to compact binary files; CPU and GPU clock pairs; the tool's own
intervals as events; the Python binding.

Standalone tests in `ruppell/tests/`, run on the node:
- (a) a synthetic graph (at least eight distinct kernels, scoped) replayed at
  least 1,000 times: every dispatch attributed to its node and scope;
- (b) eager and graph work mixed on two streams;
- (c) markers: every dispatch inside exactly one step;
- (d) buffers: zero drops at the highest dispatch rate 4.0 measured for a
  replay loop;
- (e) witness: per-kernel dispatch counts and total GPU time against
  `rocprofv3` on the same synthetic program.

### 4.2 Framework adapter (vLLM)

In `ruppell/adapters/vllm/`:
- a capture scope around each graph capture (graph key, and the op and module
  scope at each launch);
- `step_begin` and `step_end` around every engine step, enqueuing the markers;
- the scheduler's per-step state handed to the schema layer.

Check: on a short replay, every captured graph has a registry entry, and every
engine step has two markers.

### 4.3 Workload schema (serving)

In `ruppell/schemas/serving.yaml` and its loader: per step, running requests,
rows, context lengths, prefill or decode, graph key or eager; per replay,
aiperf's phase clock (`server_metrics_export.phase_time_ranges`), aligned
through the clock pairs. Check: the per-step running count averaged over the
measured phase matches Plan 16's census (1.23 at C8, 13.98 at C32) within the
between-replay spread.

### 4.4 Census, export, validate

- Census: GPU time per (layer, op) key for every step; full detail on a
  declared sample of steps.
- Export: Perfetto and Chrome JSON from the binary files, on the node.
- Validate: attributed share, unattributed kernels by name, registry-versus-
  replay sequence agreement, per-step kernel sum against step span, clock-pair
  residuals.

### 4.5 Correctness

Before any timed run: greedy tokens identical with the flag off and on over a
fixed prompt set; `DEPLOY_OK` for both trees; arm markers per arm (AGENTS.md
§10).

### 4.6 Validation on the workload

One C8 and one C32 seed-42 replay with `ruppell` on for the whole measured
phase; fresh server, discarded warmup (AGENTS.md §9.2); same node:
- coverage and the self-validation block (§5);
- agreement with an independent witness on the same work: `rocprofv3` kernel
  trace over one window of the same run, or at the same point on a separate
  replay if the two tools cannot load together (`intake decide`);
- the indexer kernels of defect 3 attributed to steps;
- reproduction of a known fact, reported as agree or disagree and not gated:
  Plan 19's ranking of cost centres by regime.

### 4.7 Overhead

Paired per-request A/B, `ruppell` off against on (always-on census mode), same
node, fresh server per point, arms alternating: at C8 six points
(OFF ON OFF ON OFF ON), then at C32 four points (OFF ON OFF ON). Per point,
record duration, successes and cache hit rate (AGENTS.md §9.1). Contrast per
bucket with `seedstudy.py` / `t18_pairs.py`. Record every host-side block over
100 ms in the ON points and its cause.

## 5. Gates

Declared now.

- **Feasibility (4.0).** Graph-launched kernels produce dispatch records with
  GPU timestamps, and a join mode is available (node ids, or an identical
  replay sequence in every replay checked).
- **Runtime (4.1).** Tests (a)–(d) pass: 100% of synthetic dispatches
  attributed, every dispatch inside exactly one step, zero drops. Test (e):
  dispatch counts equal to the witness, total GPU time per kernel within 3%.
- **Registry (4.6).** Registry and replay sequences match for at least 99% of
  replays of each graph.
- **Coverage (4.6).** At least 95% of GPU kernel time in decode steps
  attributed to a (layer, op) key, at C8 and at C32, over the whole measured
  phase.
- **Agreement (4.6).** Per kernel family, totals within 3% of the witness over
  the same work.
- **Overhead (4.7).** In always-on mode, paired per-request median ITL change
  within the OFF arm's own spread in every bucket at C8, and no
  `ruppell`-caused host block over 100 ms. C32 reported with its regime, and
  gated the same way if its four points complete.
- **Output.** The Perfetto UI and TraceLens open the exported traces.
- **Correctness (4.5).** Greedy tokens identical, flag off and on.

The settlement names every gate as passed, failed or not run, with its
artifact, and the `ruppell` commit and library digest each result came from.

## 6. Stop conditions

- 4.0 finds no dispatch records for graph-launched kernels, or no join mode:
  stop, report exactly what the SDK does and does not provide; a deeper
  mechanism is the operator's decision.
- 4.5 fails and the cause is in the hooks and cannot be repaired without
  changing served behaviour: stop.
- Node lost: keep what is measured, report what is missing, ask for a node; do
  not settle.

## 7. Artifacts

```text
$HOME/runner-ws/artifacts/23_ruppell/
<fork>/ruppell (code; every result cites a commit)
```

`FEASIBILITY.md`; test results for 4.1 (a)–(e); registries; per-step census for
every validated replay; exported traces for sampled steps (raw binary files
stay node-local and are listed by path and size); validation and agreement
tables; the overhead contrast; the settlement and `REPORT.md` (§8.3).

## 8. Execution setup

Draft, completed by the operator at submission (notebook §2.3): name the node
as "whatever node the mesh holds when it starts, checked first".

### 8.1 Mesh, node and trees

- **Mesh.** A local mesh of its own, or a queued item on an existing one.
- **Node.** One MI355X node. 4.0 and 4.1 run standalone programs: any GPU
  while no server runs, GPUs 4–7 only while one does. The server runs on GPUs
  0–3. Node time: my estimate is about a day of mostly short GPU sessions for
  4.0–4.4, then 10–12 hours for 4.5–4.7; a second allocation is likely.
- **Container.** `dsv41flash_arbor`, image by digest, weights at
  `/models/DeepSeek-V4.1-Flash`, aiperf at `/runtime/aiperf-src`. `ruppell` is
  built inside the container against the image's ROCm.
- **Trees.** `ruppell`: a working tree under `runner-ws/trees/ruppell-p23`.
  vLLM, if the adapter needs hook commits: `runner-ws/trees/vllm-p23` on a
  branch from `8f400a0ac`. Git runs on Cluster B.
- **Repository access (to settle at submission).** `<fork>/ruppell` is
  private and empty. The mesh host (WSL) has no GitHub credentials (notebook
  §2.2), and code moves by git (AGENTS.md §6). Decide where pushes to
  `ruppell` happen, and add it to the runner as a submodule pinned in
  `sources.lock.json`.
- **Not ours.** Every other mesh's jobs, nodes, sessions and artifacts.

### 8.2 Existing work to reuse (read-only; copy, never link)

- Plan 19: `mxfp8_posjoin.py` (position join and its preconditions),
  `t02_ctx.py` and the phase-clock tools, `boot_profiled_server.sh`,
  `step_attrib.py` (the `ProfilerStep#` attribution to compare against),
  `kwitness.py` (an independent parse).
- D02: boot scripts; dispatch findings (`FULL_DECODE_ONLY`, PIECEWISE hooks
  that never fire).
- D04: the frozen stall detector, for 4.7's host blocks.
- Plan 20: `t18_pairs.py` and the point driver, for 4.7.

### 8.3 The report

`REPORT.md` as the template says.

### 8.4 Things that have already cost time here

- Traces are large: parse on the node, never on a login node or through the
  mount; stage summaries.
- A capture timer is not a phase clock: place any comparison on aiperf's phase
  clock.
- Each trace has its own time base; align through clock pairs and refuse on
  mismatch (Plan 19 read a base as absolute and came out 5,909,202 s off).
- Count steps by distinct index, never by lines (Plan 19's 704 versus 200).
- Stop every server you start; `flock` and `DONE` on every point.
