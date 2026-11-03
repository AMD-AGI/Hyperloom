---
name: profiler
description: Profile a GPU workload on ROCm (LLM serving or training) to find where GPU time goes. Trace it with ruppell, vendored here, which attributes every kernel (graph replays included) to its module, op and step, instead of the PyTorch profiler; collect hardware counters with rocprofv3; read windows with TraceLens. Use when asked to profile, trace, collect counters, find bottlenecks or rank optimization sites; when a dossier needs its profile or kernel census; or when a framework or workload has no ruppell adapter yet.
---

# Profiling on ROCm: ruppell traces, rocprofv3 counters, TraceLens reports

Audience: an agent (or a person) asked where a workload's GPU time goes and
what to optimize. Read this file, then ruppell's own skill for your target
(below). Counters are in [counters.md](counters.md); the vendored copy's
provenance is in [VENDOR.md](VENDOR.md).

In an Arbor campaign, a pack opts in with `skills: [profiler]` in its
`pack.yaml`, or gets it by default if it is on Arbor's default-on list
(`DEFAULT_ON_SKILLS` in `arbor/pack/loader.py`) and the pack does not list it
in `opt-out-skills:`. Every profile the campaign takes then follows this
skill, ahead of any profiling procedure in the pack's knowledge. A pack
without it keeps its own way of profiling.

## Why ruppell, not the PyTorch profiler

On graph-replayed workloads the PyTorch profiler (Kineto) misleads. Measured on
DeepSeek V4.1 Flash serving ([ruppell/docs/background.md](ruppell/docs/background.md)):

- every kernel of a graph replay shares the replay's one `hipGraphLaunch`
  correlation id, so call sites and shapes are gone;
- steps are CPU ranges the GPU work outlives: 0 of 10,874 indexer kernels fell
  inside a GPU `ProfilerStep#` span;
- the trace export runs inside the server, and once blocked it for about 19 s.

`/hyperloom-profile` avoids the first by serving with `--enforce-eager`, which
profiles a program production does not run (no graphs, other launches).
ruppell traces the served program with graphs on, attributes every replayed
kernel to the scope that captured it, defines steps on the GPU timeline, and
checks its records against `rocprofv3` dispatch by dispatch.

> **ROCm 7 now; revisit at ROCm 10.** On ROCm 7 the profiler SDK tags a
> replay's dispatches with the launch's correlation id but not with their
> graph node, so ruppell joins each replay to its capture by position and
> refuses a replay that does not match. ROCm 10's rocprofiler-sdk puts
> `graph_exec_id` and `graph_node_id` on every dispatch from a graph launch,
> and the PyTorch trace will probably improve on top of it (Kineto PR #1567;
> PyTorch 2.15's capture-time annotations, CUDA 13.1 only so far). Our images
> are ROCm 7, so we stay with ruppell. When an image with the ROCm 10 SDK
> becomes the baseline, re-evaluate: [ruppell/docs/upstream.md](ruppell/docs/upstream.md)
> says what could then shrink (ruppell's runtime layer) and what stays (the
> adapters, schemas, census and checks).

## Which tool answers what

| Question | Tool | Never from |
| --- | --- | --- |
| Where GPU time goes per step, by module, op, kernel and step kind, graph replays included | ruppell: census, `profile_report.py` | the PyTorch profiler; an eager-mode run |
| Which kernels run, how often, at which shapes | ruppell `by_kernel`, `by_module_op`; `by_aten_op` with the CPU op tree | kernel names alone: many call sites share one symbol |
| Why a kernel is slow: bytes moved, cache, occupancy | `rocprofv3` counters at the shapes the census found ([counters.md](counters.md)) | ruppell, which records no counters |
| A window read in detail: kernel summary, ops by shape, collectives across ranks | TraceLens on ruppell's TraceLens export | TraceLens on a raw PyTorch trace of a graph-replayed server |
| Latency, throughput, host-side gaps | untraced runs (the benchmark) | any traced run: tracing slows the host |

## Get ruppell where the workload runs

`ruppell/` is ruppell at the commit in [VENDOR.md](VENDOR.md). Never edit it.
Build it on the machine, or in the container, that runs the workload:

```bash
docker cp <this skill>/ruppell <container>:/ruppell   # for a containerized server
bash /ruppell/runtime/build.sh           # ROCm with the profiler SDK (7.0.2 tested): libruppell_tool.so, libruppell_marker.so
export PYTHONPATH=/ruppell/python        # or pip install -e /ruppell, which also registers the vLLM plugin
bash /ruppell/runtime/build_cpu_ops.sh   # only for RUPPELL_CPU_OPS (aten ops), against the torch the workload runs
```

**Before every trace, check that the build is of this copy:**

```bash
cmp -s /ruppell/.commit "${RUPPELL_BUILD:-/ruppell/runtime/build}/COMMIT" || echo "stale build: rerun both build scripts"
```

- **Why:** `build.sh` writes the commit it built to `$RUPPELL_BUILD/COMMIT`, and the copy
  carries its own in `.commit`. When this copy is re-vendored, a build already in a running
  container is older than its Python side.
- **The danger:** the Python side then traces without saying so, skipping what the old tool
  lacks. Since 0.8.0 that means:
  - memory copies and fills (`copies.bin`);
  - ROCTX ranges (`RUPPELL_NVTX=1`);
  - collective calls (`calls.bin`).
- **When they differ:** rerun `build.sh` (and `build_cpu_ops.sh` if you use aten ops) with the
  same `RUPPELL_BUILD`, then check again.

On CUDA (a reference platform such as B200),
`CUDA_PATH=/usr/local/cuda-13.0 bash runtime/build.sh` builds the CUPTI tool;
everything after is the same.

## Route by target

Follow ruppell's skill, vendored with it:
[ruppell/.claude/skills/ruppell-profile/SKILL.md](ruppell/.claude/skills/ruppell-profile/SKILL.md).
It has the routing table, ten recommendations that hold for any target, and
the workflow; [reference.md](ruppell/.claude/skills/ruppell-profile/reference.md)
has the commands, checks, outputs, limitations, TraceLens use and the report
template. Where things stand at the vendored commit:

| Target | State | Do |
| --- | --- | --- |
| vLLM v2 runner `8f400a0ac`, DeepSeek V4.1 Flash, TP4, AgentX C8 and C32, seed 42, MI350 (Cluster A) | a profiling mode that passed its gates | [targets/vllm-dsv41-rocm.md](ruppell/.claude/skills/ruppell-profile/targets/vllm-dsv41-rocm.md) |
| the same stack on MI355X (Cluster B) | traced there in the runner's Plan 31: C8 and C32, every mixed step traced, decode at 5/1000 (jobs 194612 and 194611, 2026-10-05; `tests/m2/lib.sh` has a Cluster B profile); the operating-point gate has not run on MI355X | [new-target.md](ruppell/.claude/skills/ruppell-profile/new-target.md), "Same stack, different load"; say beside each number that the gate has not run there |
| the same stack, another load | the adapter works; gates not run for that load | new-target.md, "Same stack, different load"; label every number unvalidated until they have run |
| several axes at once: another model, framework build, load or parallelism (Model M on MI350X at TP8, runner Plan M00) | nothing validated carries over: a check that passed on a target differing in any one axis says nothing here | every section of new-target.md that one of the axes calls for, all run on this target, starting with "Another model"'s facts job; label every number unvalidated until they have run |
| DLRMv4 training (TorchRec, 8 ranks, MI350X) | traced; attribution, witness and same-training checks met | `ruppell/python/ruppell/adapters/dlrm/`, [milestones/dlrmv4.md](ruppell/docs/milestones/dlrmv4.md) |
| a plain PyTorch program | the script calls `instrument`, `scope`, `capture`, `step_begin`/`step_end` | [ruppell/README.md](ruppell/README.md), "Instrument a PyTorch program" |
| vLLM on CUDA (B200) | traced and checked against Nsight Systems; detail sampling (`RUPPELL_DETAIL`) since 2026-10-20 ([feasibility-cuda.md](ruppell/docs/feasibility-cuda.md)) | new-target.md, "Another platform" |
| SGLang, ATOM, TensorRT-LLM, another training stack | no adapter | the next section |

## A framework or workload without an adapter

ruppell's runtime knows nothing of frameworks. An adapter tells it each
kernel's scope, each capture's graph key and each step's fields; a new kind
of workload may also need a schema (`ruppell/schemas/serving.json`,
`training.json`).

1. Read [ruppell/docs/adapting-frameworks.md](ruppell/docs/adapting-frameworks.md)
   (the process, what an adapter provides, what bit us in vLLM, starting
   points for SGLang) and new-target.md, "Another framework".
2. Write the adapter in a ruppell checkout, on a branch from `main`, never in
   the vendored copy. It needs offline tests, and a GPU validation run as a
   job that exits when done, recorded in `docs/milestones/`.
3. Upstream it: a pull request to ruppell, following its contribution rules.
   Once merged, re-vendor here ([VENDOR.md](VENDOR.md)).
4. Until then, `rocprofv3 --kernel-trace` gives a kernel census (names,
   counts, GPU time) without scopes or steps. Say so beside every number, and
   do not rank sites by module from it.

## Workflow

1. **Know the workload first**: what the client sends, what each
   concurrency puts in flight, the regimes, the step kinds (the dossier's
   workload part; ruppell's recommendation 8).
2. **Route and choose the mode** from the route file: detail sampling, the
   step kinds to trace in full, the CPU op tree. Pin run-time tuning
   (`RUPPELL_TUNING`) on every server you will compare.
3. **Run**: `python -m ruppell run --out RUN -- <command>`; for vLLM,
   `RUPPELL_VLLM=1` in the server's environment. Exact settings:
   reference.md, "Running by hand".
4. **Check before reading**: the section 5 checks, phase coverage, the
   sampled load against the whole run's. A failed or unrun check travels with
   every number it touches. Granularity is one of them: a `torch.compile`
   region the adapter does not scope passes coverage with its time on one
   scope and fails it; rank such a run by aten op and by kernel, never by
   module. vLLM's compiled path (`-O2`) is scoped since ruppell 0.9.0.
5. **Read** `profile_report.py`'s stratified tables per regime, rank and step
   kind; collectives by transfer, never by wait.
6. **TraceLens** on a window of steps picked from the census (below).
7. **Counters** for the kernels that matter ([counters.md](counters.md)).
8. **Report** with reference.md's template, plus a counters section. Give
   each site its Amdahl ceiling; say which numbers are measured and which
   inferred; an A/B decides, not a profile. Where it goes: below.

## Where the report goes

In an Arbor campaign, each profile gets its own directory,
`<session>/profiles/<name>/`, named for what it profiled (`c8-decode`,
`c32-regime2`):

- `PROFILE.md`: the report, from reference.md's template. Keep the
  template's `## Candidates` section: the knowledge collector's summary
  quotes it.
- Beside it, the small files a reader needs to check the report:
  `profile.json`, `census_summary.json`, each `NAME_check.json`, and the
  TraceLens CSVs with their `LIMITATIONS.md`.
- Never raw records, `census.jsonl` or traces: they stay node-local (below).

The `knowledge-collector` skill shares every directory under
`<session>/profiles/` as it is. A profile kept elsewhere is passed to it with
`--profiles <dir>`, laid out the same way. Outside a campaign, keep the same
layout wherever the project keeps its profiles.

## TraceLens

```bash
python -m ruppell tracelens RUN --steps A:B -o DIR      # A:B: a regime's step_range from profile.json; one file per rank
TraceLens_generate_perf_report_pytorch --profile_json_path DIR/rank0.json --output_csvs_dir OUT \
    --include_unlinked_kernels --enable_kernel_summary --disable_coll_analysis --short_kernel_study
TraceLens_generate_multi_rank_collective_report_pytorch --trace_pattern "DIR/rank*.json" --world_size N --output_csvs_dir OUT_coll
```

- **Which reports hold** on these files, and which do not (the GPU timeline,
  category summaries when most time is `other`, roofline columns without an
  arch file, Excel output): reference.md, "TraceLens".
- **Two runs:** compare their CSVs. TraceLens's report comparison reads its
  Excel reports, and its Excel writer fails on these files.
- `LIMITATIONS.md`, written beside the TraceLens files, travels with any
  number taken from them; TraceLens's own reports drop that metadata.
- **Versions.** Arbor's `install.sh` installs TraceLens `c3405111` into the
  container; ruppell's TraceLens files were checked against `e34b294`
  (2026-09-25). If a report disagrees with reference.md, check which one ran.
- `arbor/knowledge/tracelens-usage.md` captures with `vllm serve
  --profiler-config`, the PyTorch-profiler path this skill replaces. Its
  roofline and WAIT/VMEM reading needs aten-keyed ops and an arch file for
  the GPU; ruppell's files have aten ops only with the CPU op tree, and no
  MI350 or MI355X arch file exists yet.

## On a cluster

- **Analyse on the GPU node, never on a login node.** Analysis holds a
  process's records in memory: 6.4 GB per rank per 18 minutes of C8 serving,
  on disk.
- **Raw records stay node-local.** Bring back the census summary, the profile
  report, the TraceLens CSVs with `LIMITATIONS.md`, and any window a
  conclusion cites (in a campaign, to `<session>/profiles/<name>/`).
- **Nothing else runs on the node's GPUs** while a traced or counter run is
  measuring, including GPUs the workload does not use.
- **Record the platform**: GPU and arch, ROCm and profiler SDK versions,
  image digest, ruppell commit (`run.json` keeps library digests). ruppell's
  gates ran on MI350 (SDK 1.0.1 and 1.3.2). On MI355X it has traced DSV4.1
  serving (Plan 31) without its section 5 checks or its gates (ruppell's
  [docs/limitations.md](ruppell/docs/limitations.md), "Platform"). Other ROCm versions are untested.
- **Stop every server you start.**
