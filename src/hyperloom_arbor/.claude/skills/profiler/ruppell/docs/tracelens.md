# Reading ruppell's traces with AMD TraceLens

[TraceLens](https://github.com/AMD-AGI/TraceLens) builds its analysis (op
tables, per-shape roll-ups, op categories, kernel summary, GPU timeline) from
PyTorch profiler traces. It found nothing to build on in ruppell's Chrome
trace: it reported "GPU-only trace" and gave a timeline and a kernel summary
only. So `ruppell tracelens RUN -o DIR` writes every rank of a run in the
PyTorch profiler's vocabulary, one file each, and `ruppell export --format
torch --process N` one rank (`python/ruppell/export/torch_profiler.py`):

- each kernel's ruppell scope becomes the `cpu_op` that launched it, named
  after the scope's op (the module class under the vLLM adapter; the
  default of `export --format torch`) or its module path with layer indices
  folded (`--op-names path`, the default of `tracelens`), with `Input Dims`
  from the scope's shape and the module path in `args.ruppell_scope`; with
  the CPU op tree ([cpu-ops.md](cpu-ops.md)), the recorded aten op;
- engine steps become `gpu_user_annotation` ranges;
- eager kernels keep their real launching call and its CPU interval;
- a graph replay's `cudaGraphLaunch` interval is divided among its kernels in
  node order, one op and one launch per kernel. Those CPU intervals are
  synthetic; the GPU times and the attribution are ruppell's.

A PyTorch profiler trace of the same replay gives TraceLens one
`cudaGraphLaunch` with every replayed kernel under it; TraceLens's own triage
flags that (`HIPGRAPH_LAUNCH_IN_REPORT`) and its inference report asks for a
separate graph-capture folder to recover it. Here the replayed kernels arrive
already attributed.

```bash
python -m ruppell export RUN --format torch --process 0 --steps 402:422 -o rank0.json
TraceLens_generate_perf_report_pytorch --profile_json_path rank0.json --output_csvs_dir out \
    --include_unlinked_kernels --enable_kernel_summary --disable_coll_analysis --short_kernel_study
```

Checked with TraceLens `0.1.0.dev20260930+ge34b294` on the C3 PyTorch demo
and on C4's MoE run ([`milestones/cuda-c4.md`](milestones/cuda-c4.md)): no
kernel unlinked, every replayed kernel under its module's op.

## What TraceLens does not know

- **Op categories and roofline models** key on aten op names (`aten::mm`,
  `aten::addmm`, …); ruppell's ops are modules (`RowParallelLinear`,
  `MoERunner`), so every op lands in `other` or `elementwise` and no
  roofline is computed. TraceLens also bundles GPU models for MI300X,
  MI325X and Radeon only; no B200 file was written, for want of measured
  achievable numbers.
- **Communication** is recognized by NCCL/RCCL kernel names. vLLM on B200
  all-reduces with FlashInfer's `trtllm_mnnvl_allreduce` kernels, which
  TraceLens counts as computation: on the MoE window it reported exposed
  communication at 0.26% of the time, where those kernels take 12.3% of
  kernel time.

## What the MoE trace shows

Qwen-AgentWorld-35B-A3B at TP2 on B200 (vLLM 0.30.1rc1, breakable CUDA
graphs, no `torch.compile`), rank 0, 21 engine steps from the middle of the
load phase (8 concurrent requests, mostly decode): 184.6 ms, 179.7 ms of
kernel time, 47,129 kernels.

| TraceLens table, with ruppell's scopes | |
| --- | --- |
| GPU timeline | busy 83.1%, idle 16.9% |
| Top ops by kernel time | `GemmaRMSNorm` 27.4%, `MoERunner` 23.8%, `RowParallelLinear` 13.5%, `MergedColumnParallelLinear` 7.5%, `QwenGatedDeltaNetAttention` 6.1% |
| Short kernels | 96% of kernels under 10 µs, 74% of kernel time |

Leads, from those tables and the kernels behind them:

1. **The RMS norms are unfused.** Each of the 81 norm scopes in a step runs
   13 generic PyTorch kernels (a float cast, a mean reduction, multiplies,
   a bf16 cast, an add) instead of one fused kernel: 27% of kernel time,
   more than the MoE experts. With no `torch.compile` in this
   configuration, nothing fuses them; a fused norm kernel is the largest
   single saving the trace points at.
2. **One rank waits for the other in mixed steps.** The 1,482 all-reduces
   inside the decode graphs (stream 9866) take at most 10 µs each; the 219 on
   the stream that runs mixed and prefill batches (stream 17) take 14.9 ms
   together, up to 2.4 ms each. A one-shot all-reduce that long is a rank
   arriving late, not bandwidth: skew between the ranks in those steps.
3. **The graphs fork.** About 80 graph-internal streams carry 6–7 kernels
   each per window; they are attributed like the rest (the node-id join does
   not depend on order).

These are one window of one rank of a short run, in a configuration chosen
for tracing (breakable graphs); they are leads for the workload's owners,
not measurements of vLLM's default configuration.

## Every rank, collectives, and comparisons

`ruppell tracelens` writes one file per rank, with the scope's module path as
the op name (layer indices folded, so the layers aggregate; `--op-names op`
names it by module class instead):

```bash
python -m ruppell tracelens RUN --steps 12452:12472 -o RUN/tracelens   # rank0.json, rank1.json, ...
TraceLens_generate_perf_report_pytorch --profile_json_path RUN/tracelens/rank0.json \
    --output_xlsx_path rank0.xlsx --output_csvs_dir rank0_csv
TraceLens_generate_multi_rank_collective_report_pytorch \
    --trace_pattern "RUN/tracelens/rank*.json" --world_size 4 --output_csvs_dir coll_csv
TraceLens_compare_perf_reports_pytorch base.xlsx candidate.xlsx --names base candidate -o cmp.xlsx
```

An agent (or a person) profiling a server should follow the
[`ruppell-profile` skill](../.claude/skills/ruppell-profile/SKILL.md): the
modes that passed the gates, the checks to read first, how to reconcile each
limitation, how to use TraceLens's reports, and how to rank sites. birdeye
([birdeye.md](birdeye.md)) writes the same tables without TraceLens.

## Windows, not whole runs

A full replay is 37 million kernels per rank; no Chrome-trace viewer or
JSON-based analysis holds that. TraceLens analyses steady-state windows by
design (its inference guide splits serving traces into windows, and a single
rank's window is its unit), so a window of engine steps is its expected input.
What a window cannot show, the census covers: every step of the run, per rank,
with its load (running requests, rows, context lengths), GPU time by scope and
op, busy time and period, and the client's phase. Pick the window from the
census (the steps of the regime you care about, for example C32's second
regime), export it, and read it in TraceLens. On the M2 replay: 21 decode
steps, 45 MB per rank, TraceLens's report in 24 s.

## TraceLens's Analysis agent

Not used: it is an LLM workflow on top of the reports above, and its route
for graph-mode vLLM (a capture trace from a patched vLLM, merged by batch
size and kernel name) is a workaround for what ruppell records directly. Its
approach and limits on DSV4.1: [related-tools.md](related-tools.md).
