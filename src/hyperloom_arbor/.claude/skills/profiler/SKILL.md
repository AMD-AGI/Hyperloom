---
name: profiler
description: Profile an LLM serving workload on ROCm to find where GPU time goes, with TraceLens on PyTorch-profiler traces of the served program. Start from the host's profile when there is one (in component mode, MB's TraceLens profile of the root); take a trace of your own only for what it does not cover, or after a KEEP moved the tree. Use when asked to profile, rank optimization sites, compare two trees' profiles, or write a dossier's profile (part 7) or kernel usage (part 8).
---

# Profiling on ROCm with TraceLens

Audience: an agent (or a person) asked where a serving workload's GPU time goes and what to optimize.

TraceLens (AMD-AGI/TraceLens) reads a PyTorch-profiler trace into tables: GPU time by op and by kernel, the GPU
timeline (busy, idle, communication), collectives across ranks, and rooflines. It is installed with the environment
(`$TRACELENS_ROOT`; its commands are on `PATH`).

## 1. The host's profile first

In component mode the host may have profiled the tree already. MB does: PRELUDE profiles the root with TraceLens and
gives the result to Arbor as `component.profile`, which the harness reads into your program at every iteration
("The host's profile").

- **What it holds:** `{"unit": "percent", "shares": {<name>: <share>}}`, with its source. The names are ops when
  TraceLens could attribute kernels to ops, and kernel names when it could not (a graph-replayed trace, section 4).
- **Use it:** cite it for where time goes in the root, and rank work by it.
- **Take your own trace** only for what it does not cover: a tree a KEEP has moved, a phase it did not separate
  (prefill against decode), a shape it did not reach, or op attribution it did not have.

## 2. Take a trace

A trace is taken from the running server, around a window of the measured load.

| Framework | Turn the profiler on at launch | Then |
| --- | --- | --- |
| vLLM | server flags `--profiler-config.profiler torch --profiler-config.torch_profiler_dir DIR` (older vLLM: `VLLM_TORCH_PROFILER_DIR=DIR` in the env; without either, `/start_profile` answers 404) | `POST /start_profile`, the window, `POST /stop_profile` |
| SGLang | `SGLANG_TORCH_PROFILER_DIR=DIR` in the env | the same two requests |

- **The window:** start it after the bench's warmup, during the measured load, and keep it short (seconds, or a few
  hundred steps): a trace grows fast and the profiler slows the steps it records. Record the concurrency, how many
  requests were running in the window, and whether graphs were on.
- **Where:** `DIR` is in the session (`<session>/profiles/<name>/trace/`) or node-local scratch; each rank writes its
  own trace file.
- **In a campaign iteration:** a profile is a LEARN. The profiler settings go into `arbor/config.json` for that
  iteration only and come out before `DONE`: a KEEP never carries them (a host imports a KEEP's config). Stop any server
  you started with them.

## 3. Read it

```bash
OUT=<session>/profiles/<name>
# One rank's report: ops, kernels (--enable_kernel_summary) and the GPU timeline, as CSVs (and a workbook to compare).
TraceLens_generate_perf_report_pytorch_inference --profile_json_path TRACE.json.gz \
    --output_csvs_dir $OUT/tracelens --output_xlsx_path $OUT/report.xlsx --enable_kernel_summary
# Collectives across the ranks of a TP or EP group.
TraceLens_generate_multi_rank_collective_report_pytorch --trace_pattern "DIR/*rank*.json.gz" --world_size N \
    --output_csvs_dir $OUT/collectives
# Two trees (before and after a change), from their workbooks.
TraceLens_compare_perf_reports_pytorch BASE/report.xlsx CAND/report.xlsx --names base cand -o $OUT/compare.xlsx
```

- `--group_by_parent_module` groups kernels by the module that launched them as well as by op.
- `TraceLens_split_trace` cuts a trace into steps, to read prefill and decode, or the steady state, apart.
- Each command's `--help` has the rest (rooflines, short-kernel study, extensions).

## 4. Graph replay

With CUDA or HIP graphs on (the served default), a step's kernels are replayed from the graph and the trace no
longer links them to the ops that launched them.

- **Kernel time stays right:** the kernel summary and the GPU timeline are the served program's. Rank by kernels, and
  map kernel families to ops yourself (GEMM libraries, attention, MoE, collectives, norms).
- **Op attribution needs an eager trace** (`--enforce-eager` for vLLM, `--disable-cuda-graph` for SGLang). Label it
  eager wherever it is cited: it says which op launches which kernel and ranks the ops, but its times are not the
  served ones (launch overhead is back, fusions graph capture allows may be gone).
- This is why MB's profile of a graph-replayed root holds kernel shares, not op shares.

## 5. Why a kernel is slow

The profile says where time goes; counters say why (bytes moved, cache hits, occupancy). Take them with `rocprofv3`
on the kernel alone, at the shapes the profile found, in a standalone harness (the `kernels` skill's `kbench.py`),
never on the server while it is measured.

## 6. The report

Each profile in its own directory, `<session>/profiles/<name>/`, named for what it profiled (`c64-decode`,
`after-keep-3`):

- `PROFILE.md`, the report:
  - **The window:** tree (commit), framework, flags, graphs on or eager, concurrency, running requests, steps, ranks.
  - **Where time goes:** the top ops or kernels with their share of GPU time, and idle and communication time.
  - **Against the host's profile,** when there is one: what moved.
  - **`## Candidates`:** the optimization sites it ranks, each with its share and its source table. The knowledge
    collector quotes this section.
- The TraceLens CSVs (and workbook) it cites, beside it.
- Never the raw traces in the report directory when they are large: they stay node-local, and the report names
  where.

A dossier's part 7 is such a report, and part 8 reads its kernel summary.
