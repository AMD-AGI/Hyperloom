# Status: what has been verified

Each entry names the record that holds its evidence. All the numbers on one page:
[`validation.md`](validation.md). What is still open:
[`limitations.md`](limitations.md) and [`TODOs.md`](../TODOs.md).

## M1: a Chrome trace generator on ROCm (done 2026-09-29)

On a HIP test program and a PyTorch model captured as a HIP graph, every kernel,
replayed or eager, carries its scope and its step; the trace opens in Perfetto
and `chrome://tracing`; `rocprofv3`, loaded in the same process, sees identical
dispatches. Record: [`milestones/m1.md`](milestones/m1.md).

## M2: vLLM serving (done 2026-09-29)

vLLM on DeepSeek V4.1 Flash at TP4, a C8 seed-42 AgentX replay, traced through
a vLLM plugin with no change to vLLM.
- All 40,126 graph replays per rank joined to their captures (one repaired after
  the profiler SDK mis-correlated it).
- 99.99% of GPU time attributed.
- 19,964 engine steps per rank, with the serving fields and fingerprints agreeing
  across ranks, and every kernel in exactly one step.
- `rocprofv3` identical per dispatch on every rank; windowed traces open in both
  viewers.
- Tracing leaves the computation alone: greedy outputs are identical on a small
  model, and on DSV4.1, whose outputs vary run to run even untraced, every
  checked step's GPU program (kernels, order, grids) is identical traced and
  untraced on every rank.

Record and what is still open: [`milestones/m2.md`](milestones/m2.md). The
traces read in TraceLens ([`tracelens.md`](tracelens.md)); what they show on that
replay: [`analysis/dsv41-c8-tracelens.md`](analysis/dsv41-c8-tracelens.md). M3,
the plan's gates (overhead, coverage and agreement at C8 and C32):
[`milestones/m3.md`](milestones/m3.md). Adapting another framework:
[`adapting-frameworks.md`](adapting-frameworks.md).

## CUDA on B200 (2026-09-30)

The runtime on CUPTI ([`plan-cuda.md`](plan-cuda.md)). CUPTI names the graph
node behind every replayed kernel, so replays join by node id, including graphs
that fork onto several streams.
- On a CUDA test program and the PyTorch demo every kernel carries its scope and
  its step, and Nsight Systems, on separate runs, sees the same kernels step by
  step: [`milestones/cuda-c3.md`](milestones/cuda-c3.md).
- vLLM served through the unchanged adapter:
  [`milestones/cuda-c4.md`](milestones/cuda-c4.md).
- Detail sampling (2026-10-20): sampled steps read like full detail, and at
  5/1000 serving costs under 1% of throughput
  ([`feasibility-cuda.md`](feasibility-cuda.md)).
- Against Nsight Systems and torch.profiler on four workloads:
  [`milestones/accuracy-study.md`](milestones/accuracy-study.md).

## A second workload: DLRMv4 training (2026-10-06, MI350X)

MLPerf DLRMv4, an HSTU recommender on TorchRec, trained on 8 ranks, is traced
through a training adapter (`adapters/dlrm/`), with no change to the training
code.
- It differs from M2 almost everywhere: training rather than serving, eager
  rather than graph-replayed, an autograd backward, a host running 0.6 of a
  step ahead, forked workers, kernels compiled in the process, and a ROCm
  runtime bundled in torch's wheel.
- **On 200-step runs:**
  - 99.6–99.9% of GPU time attributed;
  - 99.99% of the backward's GPU time attributed to the forward module it
    differentiates, or to DDP's gradient sync;
  - every step's kernels in exactly one step, with fingerprints agreeing
    across ranks;
  - `rocprofv3` identical per dispatch on every rank (on the system runtime:
    `rocprofv3` cannot run against torch's bundled one);
  - training metrics unchanged;
  - +1.12% step time at full detail with the CPU op tree.
- **Record:** [`milestones/dlrmv4.md`](milestones/dlrmv4.md).
- **What the traces show** (the FP8 forward's and backward's mechanisms, the
  44 ms of the backward left in norms and elementwise work, and why the
  PyTorch profiler cannot bound a step here):
  [`analysis/dlrmv4-training-profile.md`](analysis/dlrmv4-training-profile.md).

## M6: memory copies and fills (2026-10-20, MI350X and B200)

Copies and fills are recorded in `copies.bin` on both runtimes (roadmap-v2 M6), and they are
attributed to scopes and steps like kernels.
- **MI350X:** `rocprofv3 --memory-copy-trace`, in the same process, saw the same copies with
  identical times: 15,627 on Qwen3-8B, 38,381 per rank on Qwen3-30B-A3B.
- **B200:** Nsight Systems, on a separate run, counted the same copies of every kind.

Record: [`roadmap-v2.md`](roadmap-v2.md) M6; the numbers:
[`validation.md`](validation.md) §6.

## M7: NVTX and ROCTX ranges (2026-10-20, MI350X and B200)

With `RUPPELL_NVTX=1`, the program's own NVTX (CUDA) and ROCTX (ROCm) ranges are recorded in
`ranges.bin`. Each kernel takes the range open on its launching thread at launch; a replayed
kernel takes the one open at capture.
- **Against the CPU op tree:** every `emit_nvtx` op range named the same op, eager and replayed.
- **Against the vendor:** rocprofv3's marker trace (MI350X, same process) has the same 768 ranges,
  and so does nsys (B200, separate run).
- **Served:** vLLM's layerwise ranges and NCCL's arrive.

Record: [`roadmap-v2.md`](roadmap-v2.md) M7; the numbers: [`validation.md`](validation.md) §7.

## M8: collectives (2026-10-21, MI350X and B200)

Each collective kernel carries its operation, dtype, size and group, from two sources, both on by
default:
- **The vLLM adapter** wraps vLLM's communicators and backends (custom all-reduce, symmetric
  memory, pynccl, torch). Every collective kernel but 1–2 startup all-reduces per rank has its
  call, graph replays included, at TP2, TP4 and TP8.
- **ruppell's NCCL profiler plugin** (`nccl.tsv`; interface v5 for NCCL, v4 for RCCL) gives the
  library's own record of each call it runs: function, count, datatype, communicator, algorithm,
  protocol, channels and their GPU span. That covers programs without an adapter, such as
  training.

What was checked:
- **Sizes:** every adapter call's bytes equal the library's record.
- **Counts:** equal to Nsight Systems' (B200, separate run) and `rocprofv3 --rccl-trace`'s (MI350X,
  same process).
- **Bandwidth:** birdeye's `coll_by_op` reports it per call kind, against nccl-tests and
  rccl-tests.
- **Training:** DDP on both platforms and DLRMv4 on MI350X, from the library's record alone.
  RCCL's MSCCL kernels are the exception: RCCL reports nothing for them.

Record: [`roadmap-v2.md`](roadmap-v2.md) M8; the numbers: [`validation.md`](validation.md) §8.

## M9 gray-parrot on CUDA: counters per step (2026-10-22, B200)

Hardware counters, off by default ([`gray-parrot/`](gray-parrot/README.md)). On CUDA each counter
step is one CUPTI Range Profiler user range: per kernel is out on CUPTI 130001, whose auto range
skipped CUDA graph launches.
- **Coverage:** every counter step had its reading, with graph replays inside: ruppell's demo,
  Qwen3-8B served, Qwen3-30B-A3B at TP2 (both ranks counted the same steps), and a training step.
- **Against Nsight Compute,** on the demo: warps launched equal, instructions +3.6%.
- **Timing steps untouched, off is off,** and outputs within the program's own run-to-run spread.
- **Cost:** 60–120 ms of host time per counter step, so the CUDA default is 1 step in 1,000.
- **The sets (GP3 on sm_100):** tcgen05's tensor-op counters read exactly 2·M·N·K on plain GEMMs,
  padding included where a kernel pads. A Qwen3-8B decode step's DRAM read equals its weights and
  KV cache within 0.5%.
- **Device activity (GP4):** PM sampling at 10 kHz, evaluated in the tool.
  - **Against Nsight Systems' GPU metrics:** within a few percent.
  - **Cost:** 0.5–1.1% of throughput and about 0.65 of a host core.

Records: the runner's `ruppell-accuracy/results/gray-parrot/gp2/`, `gp3/` and `gp4/` (plan 04,
Parts B to D).
