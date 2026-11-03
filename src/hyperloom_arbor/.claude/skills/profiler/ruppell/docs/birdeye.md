# birdeye: a built-in micro TraceLens

`ruppell birdeye` writes TraceLens's standard tables (GPU timeline, kernel
summary, ops, categories, short kernels, collectives, GEMM) and ruppell's own
step table from a ruppell run or from another profiler's trace, with numpy
and the standard library only. It exists so a report needs neither
TraceLens's install nor its Excel writer, reads graph-replayed kernels with
their ops already attached, and can put a ruppell run beside an nsys or
rocprofv3 trace of the same work.

```bash
python -m ruppell birdeye RUN                       # a ruppell run: RUN/birdeye/
python -m ruppell birdeye RUN/trace.json --steps 6928:6948 --out OUT   # its Chrome trace, a window
python -m ruppell birdeye rank0.json.gz             # a PyTorch profiler (Kineto) trace
python -m ruppell birdeye W/nsys.sqlite             # nsys export --type sqlite
python -m ruppell birdeye W/                        # rocprofv3 --kernel-trace --output-format csv
python -m ruppell birdeye-compare CMP base_out cand_out --tags base,cand
```

Options: `--format auto|ruppell|nsys|rocprof|kineto` (auto by default),
`--steps A:B` (step ids), `--phase NAME=START:END` (realtime seconds for a
ruppell run, through its clock pairs; seconds from the first activity
otherwise; it splits the step tables by phase), `--rank N` (the rank of the
per-rank tables, default the lowest), `--short-us 10`, `--top 30` (rows per
table in REPORT.md), `--comm ruppell|tracelens` (the collective classifier,
below).

Outputs, in `OUT` (default `RUN/birdeye/` or `FILE_birdeye/`):

- `<sheet>.csv` per table, with TraceLens's file and column names where the
  meaning is the same;
- `birdeye.json`: provenance (the recording build, the file digest, the
  analysing commit), coverage, the tables not written and why, the
  limitations;
- `REPORT.md`: provenance, coverage, a mechanical "what stands out" list,
  then the tables.

## Sources

Every source becomes one normalized kernel table
(`python/ruppell/birdeye/readers.py`): one row per GPU activity with its rank,
device, stream, name, GPU interval, kind (kernel, memcpy, memset), step, scope,
op, op signature (input dims and types), op invocation, launch kind (eager or
graph) and whether its launching call is in the source.

| source | reader | ops | steps | launches | copies |
| --- | --- | --- | --- | --- | --- |
| ruppell run directory | `engine.analyze_run` (graph join, steps, scopes, CPU op tree, backward attribution) | scope's op (module class), or the aten op with `RUPPELL_CPU_OPS` | ruppell's (markers, step ids, fields) | yes | ROCm blit kernels; none on CUDA |
| ruppell Chrome trace (`ruppell export`) | its kernel events' `scope`, `op`, `step`, `attribution` | as recorded | ruppell's | yes (the window's API calls) | as recorded |
| PyTorch profiler / ruppell's `--format torch` files | the innermost `cpu_op` open on the thread of the runtime call that launched the kernel (same `correlation`), as TraceLens links them; TraceLens's pass-through `execute` op is skipped | `cpu_op` events, `Input Dims`, `Input type` | `ruppell_step` on kernels; else ruppell's marker kernels; else `ProfilerStep#N` GPU annotations; else `ProfilerStep#N` CPU ranges by launch time | `cuda_runtime`/`hip_runtime` events; `*GraphLaunch` is a graph | `gpu_memcpy`, `gpu_memset` |
| nsys SQLite (`--cuda-graph-trace=node`) | `validate.witness._read_nsys`, plus the memcpy and memset tables | none | ruppell's marker kernels, numbered in time order | `graphNodeId`; linked when the correlation id is in the runtime table | yes |
| rocprofv3 kernel trace CSV | `validate.witness._read_csv`, plus `*memory_copy_trace.csv` beside it | none | ruppell's marker kernels | unknown | memory copies, and rocclr copy kernels |

A source that does not name its ranks (nsys, rocprofv3, a PyTorch trace
without `distributedInfo`) gets them numbered in process order, and the
report says so. Step markers are never counted as work.

## The tables

TraceLens means its perf report, `generate_perf_report_pytorch`, at e34b294.

### `gpu_timeline` (TraceLens: `gpu_timeline`)

Columns `rank,type,time ms,percent`; types `computation_time`,
`exposed_comm_time`, `exposed_memcpy_time`, `busy_time`, `idle_time`,
`total_time`, `total_comm_time`, `total_memcpy_time`, per rank and their mean.

Each activity is communication (its name is a collective's), memcpy (a
memcpy event, or a ROCm `__amd_rocclr_copyBuffer*`/`MEMORY_COPY_*` kernel, as
TraceLens re-buckets them) or computation (everything else, fills
included). Each class is the union of its intervals; exposed communication is
communication minus computation; exposed memcpy is memcpy minus computation
and communication; busy is the union of everything; total is the last end
minus the first start; idle is total minus busy. The arithmetic is exact in
integer ns, so computation + exposed comm + exposed memcpy + idle = total.

Differences from TraceLens:
- **Collectives.** By default birdeye uses ruppell's classifier
  (`limits.COLLECTIVE`: RCCL/NCCL, MSCCL, vLLM's and AITER's custom
  all-reduce, quick reduce, FlashInfer's and TRT-LLM's one-shot and two-shot
  all-reduce kernels). TraceLens recognizes `nccl`, `rccl` and
  `cross_device_reduce` only, so it counts FlashInfer's `trtllm_mnnvl_allreduce`
  as computation. `--comm tracelens` uses TraceLens's names and reproduces its
  numbers (below).
- **Unlinked kernels.** birdeye counts every activity, as TraceLens does with
  `--include_unlinked_kernels`; without the flag TraceLens drops kernels whose
  launch it cannot find. The coverage section counts them.
- **Every rank** of a multi-process source, and their mean; TraceLens reads
  one rank per file.

### `gpu_timeline_by_step_kind` (birdeye's own)

The same timeline inside steps: each step's activities clipped to its GPU
period (begin marker to end marker), summed over the steps of each kind
(decode, prefill, mixed, a startup phase), per rank. Total is the sum of the
periods, so idle is the GPU's idle time inside steps, not between them.

### `kernel_summary` (TraceLens: `kernel_summary`)

Kernels (not copies) of the chosen rank grouped by `Parent op category`,
`Parent cpu_op`, `Kernel name` (the demangled full name) and `Kernel stream`:
`Kernel duration (µs)_sum, _count, _mean, _min, _max`, `Percent of kernels
time (%)`, `Percent of total time (%)` (of the timeline's total).

- `Parent cpu_op`: the kernel's op (the aten op with the CPU op tree or in a
  PyTorch trace; otherwise the ruppell scope's op, the module class). A
  kernel with no ruppell scope but an NVTX or ROCTX range (`RUPPELL_NVTX=1`, M7) takes
  `nvtx:<range>`, its innermost range, as its scope and op. A kernel
  with none: `graph` (graph-replayed), `runtime` (an eager launch outside any
  op), `unlinked` (its launching call is not in the source), empty (the
  source records no launches: rocprofv3).
- `Parent op category`: the op's category when its name decides it (aten and
  registered `namespace::op` GEMMs, attention, norms, MoE ops), otherwise the
  kernel's category from its name (below). TraceLens categorizes by op name
  only, so ruppell's module ops all land in `other` or `elementwise` there.

### `ops_summary` and `ops_summary_by_category` (TraceLens: same names)

`ops_summary`: `name,total_direct_kernel_time_sum,total_direct_kernel_time_ms,Count,Categories,Percentage (%),Cumulative Percentage (%)`.
`ops_summary_by_category`: `op category,Count,total_direct_kernel_time_ms,Percentage (%),Cumulative Percentage (%)`.

An op invocation's direct kernel time is the union of its kernels' intervals;
an op's time is the sum over its invocations, `Count` the number of
invocations. Collective kernels are left out, as TraceLens's kernel launchers
leave out NCCL kernels. An invocation is:
- with the CPU op tree: the aten op record that launched the kernel (eager),
  or (that op record, graph, replay) for a replayed kernel, so a cuBLAS
  split-K GEMM's two kernels count once;
- without it: the launching call (eager), or the replayed kernel itself. That
  is one op per launch as `ruppell export --format torch` writes it, so the
  table agrees with TraceLens's on those files;
- in a PyTorch trace: the `cpu_op` event, else the runtime call.

`Categories` and `op category` are birdeye's (the op's category, else the
category of the invocation's longest kernel), not TraceLens's op categorizer.
Sources without ops (nsys, rocprofv3) get no ops tables; `category_summary`
covers them.

### `category_summary` (birdeye's own)

Every activity of the chosen rank by kernel category: `category,Count,Kernel
duration (µs)_sum,busy (µs),Percent of kernels time (%),Percent of total time
(%)`. It needs no op attribution, so it is the table that compares across
vendors and profilers.

### `short_kernel_histogram`, `short_kernels_summary` (TraceLens: same names)

Kernels under `--short-us` (10 µs): a 100-bin histogram of their durations
(`bin_start,bin_end,count`, as `np.histogram`), and per `Parent cpu_op` and
`Kernel name` their `Short Kernel duration (µs) sum`, `Short Kernel count`,
`Short Kernel duration (µs) mean` and `Short Kernel duration (µs) percent of
total time`. TraceLens also groups by input dims, strides and concrete inputs
and drops a row when one is missing, so its summary comes out empty on files
without them (the agentx CSVs are).

### `coll_analysis` (TraceLens: `coll_analysis`, which differs)

TraceLens's collective analysis reads NCCL metadata (process groups, message
sizes) that only a PyTorch trace carries. birdeye has kernels only. Per
collective kernel (short name), over every rank: `Ranks`, `Count`, `Duration
(µs)_sum, _mean, _min, _max`. With several ranks, each launch is aligned
across ranks (same step, same position among the step's collectives, and the
step's sequence of collective names identical on every rank; steps that differ
are skipped) and split as `tests/m3/profile_report.py` does: `Transfer
(µs)_sum` (each launch's shortest duration across ranks), `Wait
(µs)_sum_mean_rank` (the rest, mean per rank) and `Wait (%)`; and `Start skew
(µs)_mean, _max` (the latest start minus the earliest) when the ranks share a
clock (one ruppell run on one host, one nsys file, one rocprofv3 node; not
separate PyTorch files). Waiting and skew are untrusted in a traced run
(limitation `collective-wait`).

### `coll_by_op` (no TraceLens counterpart; ruppell records only, M8)

One row per call kind, over every rank: the call's `Operation`, `Backend`, `Dtype`, `Elements`,
`Bytes` and `Group size`, from the vLLM adapter's records, or from the NCCL plugin's record of a
call no adapter recorded. Then `Ranks`, `Count`, `Graph-replayed`, `Duration (µs)_sum, _mean`;
`Aligned launches` and `Transfer (µs)_mean`, aligned as `coll_analysis` aligns them;
`Algorithmic bandwidth (GB/s)` and `Bus bandwidth (GB/s)` from the transfer, with nccl-tests'
size and bus factor per operation; `Wait (%)`; and the library's `NCCL algorithm`, `NCCL
protocol` and `NCCL channels`. With one rank there is no transfer estimate, so no bandwidth.
Against nccl-tests and rccl-tests, the run reads slower by a few µs per call, which the
libraries' own timers show too (`validation.md` §8). The table is in `REPORT.md` as "By call".

### `GEMM` (TraceLens: `GEMM`, fewer columns)

`name,param: B,param: M,param: N,param: K,param: dtype_A_B,GFLOPS_first,Data
Moved (MB)_first,FLOPS/Byte_first,TFLOPS/s_mean,TB/s_mean,Kernel Time
(µs)_sum,Kernel Time (µs)_mean,name_count`, one row per op and shape.

Ops: `aten::mm`, `addmm`, `bmm`, `baddbmm`, `matmul`, `linear` and
`_scaled_mm`, from their input dims (the innermost op owns the kernels, so
`linear` around `addmm` counts once, as `addmm`). FLOPs = 2·B·M·N·K; bytes =
(M·K + K·N + M·N)·B at each operand's element size (the output at A's); MB is
2^20 bytes and TB/s 10^12 bytes per second, as TraceLens's are. Times are each
invocation's direct kernel time. No bias, stride or transpose parameters and
no roofline (no arch file). The table needs op shapes: the CPU op tree
(`RUPPELL_CPU_OPS=1`) or a PyTorch trace recorded with shapes; without them it
is not written, and the report says why (`gemm-needs-shapes`).

### `steps` (birdeye's own; TraceLens has none)

Per rank and step kind (and phase with `--phase`): `steps`, `steps with
kernels`, `period (µs)_mean, _median, _p90` (begin marker to end marker),
`busy (µs)_mean` (the union of the step's activities inside its period),
`busy/period`, `kernels per step_mean`, `kernel time (µs) per step_mean`,
`graph kernels per step_mean`, `eager kernels per step_mean`.

## Kernel categories

`python/ruppell/birdeye/categories.py` names each kernel's category from its
demangled name with ordered rules: comm (the collective classifier), memcpy,
memset, triton (`triton_` prefix), sampling as reduce, MoE (`moe`, `fmoe`,
grouped GEMM, `topk`), attention (flash, fmha, MLA, MQA, paged attention, KV
cache writes, gated delta rule), norm (unless the name is a GEMM),
quantization as elementwise (unless a GEMM), GEMM (`gemm`, cutlass, cuBLAS,
nvjet, hipBLASLt `Cijk_`, `bmm_`, `*linear_kernel`), elementwise, reduce,
other. The order is the rule: a name with several hints takes the first, and
the module docstring says why each comes where it does. On the kernels of the
B200 artifacts (`tests/test_birdeye_offline.py`) `other` is 4.7% of kernel
time. Names cannot tell an attention indexer's top-k from MoE routing; every
top-k is MoE (`categories-by-name`).

## Compare

`ruppell birdeye-compare OUT_DIR A B ...` merges reports (or sources, reported
first into `OUT_DIR/<tag>/`), the first as the baseline, like TraceLens's
`compare_perf_reports_pytorch`: per table an outer join on its keys
(`gpu_timeline`: rank and type; `category_summary`: category; ops tables: op
or op category; `kernel_summary`: kernel name, summed over ops and streams;
`steps`: rank and step kind; `coll_analysis`: collective), columns
`tag::column`, and for each later report `tag::column__diff` and
`tag::column__pct` against the baseline. Output `compare_<table>.csv` and
`COMPARE.md`. Kernel names do not match across vendors or library versions;
the timeline, category and step tables are the ones that compare across them.

## Agreement with TraceLens

On the same input, birdeye with `--comm tracelens` reproduces TraceLens's CSVs
(`tests/test_birdeye_offline.py`, part d):

| input | table | birdeye / TraceLens |
| --- | --- | --- |
| B200 DSV4.1 AgentX C8, steps 6928–6948, rank 0 (`agentx_c8`) | gpu_timeline, all eight rows | identical to 1e-9 (computation 166.8278 ms, exposed comm 1.2397, busy 168.0675, idle 14.5583, total 182.6258) |
| | kernel_summary | 37,743 kernels, 204,333.527 µs, 445 rows: identical |
| | ops_summary | every op's direct time and invocation count identical (21 ops) |
| B200 C32, steps 6903–6923, rank 0 (`agentx_c32`) | gpu_timeline | identical (computation 186.0280 ms, busy 187.5215, total 202.2209) |
| | kernel_summary, ops_summary | 37,911 kernels, 223,466.401 µs: identical |

The TraceLens CSVs were made from rank 0 of `ruppell export --format torch`
of that window; the file itself is not kept, so birdeye reads the Chrome trace
of the same window of the same run (`trace.json`), which holds the same
kernels. With ruppell's own classifier the busy and idle times are unchanged
and FlashInfer's all-reduce moves to communication: on C8, total
communication 9.708 ms (TraceLens 1.240) and exposed 8.359 ms (1.240).

## Limitations

The ones birdeye adds are in `ruppell.limits` and in every `birdeye.json` and
REPORT.md, beside the source's own (`collective-wait`, `traced-latency`,
`module-ops`, ...):

- `comm-by-name`: collectives are recognized by kernel name;
- `categories-by-name`: kernel categories come from names;
- `gemm-needs-shapes`: the GEMM table needs op shapes;
- `memcpy-unrecorded`: ruppell on CUDA records no memcpy or memset.

Also: a ruppell run is analysed whole, in memory, before `--steps` cuts it,
as the census and the export do (a full serving run is tens of millions of
kernels per rank; for one, read a windowed export); the nsys, rocprofv3 and
PyTorch-profiler readers are tested on synthetic files only; TraceLens's
roofline, per-op perf models other than GEMM, and its inference report are not
reproduced.
