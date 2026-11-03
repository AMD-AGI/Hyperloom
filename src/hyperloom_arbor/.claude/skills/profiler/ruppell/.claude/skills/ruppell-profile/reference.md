# ruppell-profile: reference

Look-up material for [SKILL.md](SKILL.md), true for any target. Measured
values for a target are in its route file (`targets/`).

## Running by hand

```bash
bash runtime/build.sh                 # RUPPELL_BUILD=DIR to choose where; libruppell_tool.so, libruppell_marker.so (ROCm; CUDA_PATH=... for CUDA)
bash runtime/build_cpu_ops.sh         # the CPU op tree, against the torch the server runs (only with RUPPELL_CPU_OPS)
# register the vLLM plugin once per environment: entry point vllm.general_plugins
#   ruppell = ruppell.adapters.vllm:register  (pip install the package, or tests/m2/lib.sh m2_install_plugin)
RUPPELL_VLLM=1 RUPPELL_DETAIL=5/1000 RUPPELL_DETAIL_KINDS=mixed,prefill RUPPELL_CPU_OPS=1 \
RUPPELL_TUNING=/work/tuning.json \
    python -m ruppell run --out RUN -- vllm serve ... --shutdown-timeout 60   # then drive the workload (aiperf)
# --shutdown-timeout: at 0 (vLLM's default) vLLM SIGKILLs its workers on SIGINT before their last flush
python3 tests/m2/aiperf_phases.py PROFILE_EXPORT.jsonl > phases.json
AIPERF_JSONL=PROFILE_EXPORT.jsonl WINDOW=auto bash tests/m2/post_run.sh NAME RANKS none   # census, export, checks (reads /work/NAME)
python3 tests/m3/profile_report.py RUN --aiperf phases.json --split SECONDS --md profile.md > profile.json   # --split 0: one regime
python -m ruppell tracelens RUN --steps A:B -o DIR          # TraceLens files, one per rank
python -m ruppell birdeye RUN --steps A:B --out DIR         # TraceLens-style tables and REPORT.md, no TraceLens needed
```

| variable | values | what it does |
| --- | --- | --- |
| `RUPPELL_VLLM` | `1` | activates the vLLM adapter in every process |
| `RUPPELL_DETAIL` | `all`, `K/N` | kernel records for K of every N engine steps (the same steps on every rank); startup, capture and warmup always traced. Refused on CUDA (every step traced) |
| `RUPPELL_DETAIL_KINDS` | e.g. `mixed:1/5,prefill:1/5`, or `mixed` for every step | those step kinds traced at a rate of their own (counted per kind), whatever `RUPPELL_DETAIL` samples |
| `RUPPELL_CPU_OPS` | `1`, `names`, unset | the aten op tree, with input shapes (`1`) or names only |
| `RUPPELL_TUNING` | a file path | pins Inductor and Triton autotuning across ranks and servers; give every server you compare the same file |
| `RUPPELL_VLLM_COMPILED` | `0` | turns off the scopes inside vLLM's compiled regions (on by default; they set `VLLM_DISABLE_COMPILE_CACHE=1`, and `TRITON_CACHE_DIR` where Inductor would when unset) |
| `RUPPELL_HIP_API` | `analysis` (default), `all`, `none` | which HIP API calls are recorded |
| `RUPPELL_MODE` | `witness` | markers only, for running under an independent collector |

## Checks

`post_run.sh` writes `NAME_check.json` (`tests/m2/check_m2.py`), per rank:

| check | what a failure means |
| --- | --- |
| every replay joined or refused with a reason; registry ≥ 99% of each graph's replays | replayed kernels without attribution |
| coverage: decode GPU time on a (layer, op) key ≥ 95% | more than 5% of decode time in the runner or nowhere; module shares are of a smaller whole |
| granularity: decode GPU time on a layer (`decode_layer_share`) ≥ 80%, and on the largest single scope (`decode_top_scope_share`) ≤ 50% | the time is on a coarse scope, though covered: a `torch.compile` region runs no module hooks, and its kernels carry the scope around it (vLLM `-O2` before 0.9.0: `model`, 86–92% of decode). Rank by aten op and by kernel, not by module |
| running requests, census against the client's in-flight count | the steps the census saw are not the load the client drove |
| every kernel after the first engine step in exactly one step; side-stream kernels owned | per-step numbers unreliable |
| merge (fingerprints across ranks), schema, dropped records = 0 | ranks disagree, fields missing, or the trace is incomplete |
| records complete: a final flush, or the adapter's flush at shutdown (census `incomplete_processes`) | the worker was killed before its last flush and lost its last records, which `dropped records` does not count; serve vLLM with `--shutdown-timeout 60` |
| witness (when run under `rocprofv3`) | ruppell's records differ from an independent collector's |

Then from `profile_report.py`:
- **`phase_coverage`**: each step kind's share of engine time over every
  step, and its sampled steps. A kind is `undersampled` at ≥ 5% of the time
  with < 20 samples.
- **`stratified.uncovered_kinds`**: kinds with no sample.
- **`load_sampled_steps` against `load_all_steps`**: whether the sample is
  the load.

## Outputs

`profile.json`:
- `workers[i].regimes.<regime>`:
  - `engine_steps`, `sampled_steps`, `step_range`;
  - `phase_coverage` (above);
  - `load_all_steps` and `load_sampled_steps`;
  - `kernel_ms_per_sampled_step` and `busy_over_period`;
  - `by_kernel`, `by_module`, `by_module_op`, `by_aten_op`: the sampled
    steps as they are;
  - `stratified`: `scale` (steps per sample, per kind), `uncovered_kinds`,
    `kernel_ms_estimated`, and `by_kernel`, `by_module`, `by_aten_op`,
    weighted by kind. **Rank from these.**
- `collectives.<regime>`: `transfer_ms`, `transfer_by_kernel`,
  `wait_ms_per_rank` (reported only), `steps_aligned`.
- `limitations`: the ones that hold for the run (`python/ruppell/limits.py`).

Also:
- **Census:** `census.jsonl` (every traced step: fields, GPU time by op,
  busy, period, client phase) and `census_summary.json`.
- **Chrome trace:** `trace.json`, a window.
- **Host blocks:** `host_blocks.json` (ruppell-caused host blocks over 100 ms).
- **TraceLens:** `tracelens_*/` (CSVs, per-rank files, `LIMITATIONS.md`).
- **birdeye:** `birdeye/` (`REPORT.md`, `gpu_timeline.csv`, `kernel_summary.csv`,
  `ops_summary*.csv`, `category_summary.csv`, `short_kernel*.csv`,
  `coll_analysis.csv`, `steps.csv`, `GEMM.csv` with the CPU op tree,
  `birdeye.json`); `birdeye-compare` for two runs ([`docs/birdeye.md`](../../../docs/birdeye.md)).

## Reconciling the limitations

The ids are `python/ruppell/limits.py`'s, carried in every output. The route
file gives each one's measured size on its target.

| limitation | symptom | what to do |
| --- | --- | --- |
| **collective wait** (`collective-wait`) | per-rank collective time differs by large factors; the traced straggler is not the untraced one | use the transfer estimate (each launch's shortest duration across ranks); rank skew needs an untraced measurement |
| **straggler's compute shift** | one rank's compute kernels a few percent faster or slower than the others | compare ranks; differences smaller than the target's measured shift are unresolved |
| **host time of traced steps** | kinds traced at every step take a larger share of engine time than when sampled | step-time weights from a lower-rate run; kernel-time estimates (GPU time) are not inflated this way |
| **sampling** (`detail-sampled`) | totals over sampled steps; large `scale` | use `stratified`; a kind's estimate rests on its sampled steps only |
| **under-sampled kind** | `undersampled` or `uncovered_kinds` non-empty; sampled load far from the whole run's | add the kind to `RUPPELL_DETAIL_KINDS`, or state its time share as unseen |
| **runner work** | coverage under 95% | the rest is the framework's own work (`runner.*` scopes); rank it as framework overhead |
| **compiled region** | the granularity check fails; one scope (`model`) holds most of decode | a `torch.compile` region the adapter does not cover runs no module hooks (vLLM's compiled path is covered since 0.9.0; `compiled.piece` is a kernel no launch site placed); rank by aten op and by kernel, or serve with breakable graphs (`VLLM_USE_BREAKABLE_CUDAGRAPH=1`) |
| **no aten op** (`module-ops`) | `(no op)` in `by_aten_op` | kernels launched outside any op (Triton from Python); rank them by module |
| **tuning pinned** | `tuning.json`, "tuning: replayed" log lines | block sizes are the first process's choice on every rank; untraced production may differ per rank |
| **latency and throughput** (`traced-latency`) | | never from a traced run |
| **host-side analysis** | | the traced host is slower (per-dispatch tracing, op-tree callbacks); no launch-boundness or CPU-gap conclusions from a traced run |
| **TraceLens files** (`synthetic-cpu-intervals`) | | host intervals of replayed kernels are made up; the timeline counts gaps between sampled steps as idle |
| **software timestamps** (`software-timestamps`, CUDA) | short kernels read 1.1–1.3× a hardware-timestamped profile's | ruppell adds 0.2–0.5 µs per kernel against Blackwell's hardware timestamps, and matches software-timestamped nsys within 0.2%; compare durations with like-timestamped profiles |
| **unvalidated target** | no route file, or gates not run for this load | label every number unvalidated; see [new-target.md](new-target.md) |

## TraceLens

```bash
python -m ruppell tracelens RUN --steps A:B -o DIR       # A:B = a regime's step_range from profile.json
TL=~/.venvs/tracelens312/bin                              # TraceLens e34b294 on Cluster A; or pip install TraceLens
$TL/TraceLens_generate_perf_report_pytorch --profile_json_path DIR/rank0.json --output_csvs_dir OUT \
    --include_unlinked_kernels --enable_kernel_summary --disable_coll_analysis --short_kernel_study
$TL/TraceLens_generate_multi_rank_collective_report_pytorch --trace_pattern "DIR/rank*.json" --world_size N --output_csvs_dir OUT_coll
```

- **Use:**
  - `kernel_summary.csv`;
  - `ops_summary.csv`: its ops are ruppell's module paths
    (`--op-names path`, layers folded) or module classes (`--op-names op`),
    and aten ops with the CPU op tree;
  - the per-category CSVs, with shapes;
  - `short_kernel_histogram.csv`;
  - the collective report: latency against skew, with collective-wait in
    mind.
- **Or birdeye** (`python -m ruppell birdeye`): the same tables without
  TraceLens, every rank's timeline, the timeline inside steps by kind,
  kernel categories that compare across vendors, collectives split into
  transfer and waiting, and ruppell's steps. It reproduces TraceLens's
  timeline and summaries with `--comm tracelens`; by default it counts
  custom all-reduce kernels (FlashInfer, vLLM, AITER) as communication.
  The same caution on sampled runs: a window's idle time includes the gaps
  between sampled steps; `gpu_timeline_by_step_kind.csv` is inside steps.
- **Do not use:**
  - `gpu_timeline.csv`: the files hold sampled steps only;
  - `ops_summary_by_category.csv`, when most time is `other`;
  - roofline columns, without an arch file for the GPU;
  - Excel output (its writer fails on these files; CSVs only);
  - TraceLens's Analysis agent (`docs/related-tools.md`).

## Report template

```markdown
# <model> on <workload>: profile at <concurrencies>

**Run.** <cluster, node, GPUs>, <framework pin, image digest>, <seed, workload settings>;
mode <settings> (gates: <passed where, or unvalidated>); ruppell <commit>; jobs <ids>.
**Checks.** <every section 5 check: pass, or fail and why>; phase coverage per regime
(<kind: time share, sampled>); sample against load (<running, rows>).

## <concurrency>, <regime>
| rank | site (module) | stratified share | kernels | step kinds | Amdahl ceiling |
Collectives: transfer <ms> (<kernels>); waiting not used (collective-wait).

## Candidates
1. <site>: <share> per concurrency and regime; <why it is a lever>; limitations: <ids>.

## What this cannot say
<latency, rank skew, host gaps, uncovered kinds, unvalidated parts, anything a check left open>
```

## Evidence

- `docs/milestones/`: what was verified, run by run (M1–M3, CUDA C3–C4).
- `docs/plan-m3.md`: the gates and the operator's decisions.
- `docs/analysis/`: profiles written up.
- `docs/related-tools.md`: TraceLens's Analysis agent and Hyperloom's phase
  coverage.
- `docs/limitations.md` and `TODOs.md`: what is open.
