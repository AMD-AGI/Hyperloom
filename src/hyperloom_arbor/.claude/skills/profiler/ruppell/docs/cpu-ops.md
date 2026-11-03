# The CPU op tree (RUPPELL_CPU_OPS)

Off by default. With `RUPPELL_CPU_OPS=1`, ruppell also records, on every host
thread, the aten ops PyTorch runs (with their input dims and dtypes) and the
ruppell scopes as intervals, the module tree. Every kernel then carries the
aten op that launched it, graph-replayed kernels included: the op that was
running when their graph node was captured. This is what a PyTorch profiler
trace cannot give for CUDA-graph replays, and what TraceLens's op categories
and performance models need.

```bash
RUPPELL_BUILD=... bash runtime/build_cpu_ops.sh           # with the traced program's python (in its container)
RUPPELL_CPU_OPS=1 python -m ruppell run --out RUN -- ...   # or =names: op names only, no input capture
python -m ruppell export RUN --format torch --steps A:B -o rank0.json   # then TraceLens, as in tracelens.md
```

Under vLLM, `tests/cuda/serve_vllm.sh NAME TP traced_ops` builds the library
for the container's torch and serves with it.

## How it works

- **`runtime/src/cpu_ops.cpp`**, built against libtorch only (the same on
  CUDA and ROCm), registers a global `RecordFunction` callback for aten ops
  and user `record_function` scopes. It does not use `torch.profiler` or
  Kineto, and no GPU tracer: nothing competes with the tool's CUPTI or
  rocprofiler session. Op signatures (name, dims, dtypes) are interned per
  thread; the library refuses a torch other than the one it was built for.
- **The platform tool owns the clock.** The library calls
  `ruppell_op_enter` / `ruppell_op_exit`; the tool timestamps both ends in its
  own domain and writes `ops.bin` (`ruppell_op_rec`) and `ops.tsv`
  (`runtime/src/cpu_tree.h`, shared by `tool_cuda.cpp` and `tool.cpp`). The
  tool also times each scope from `ruppell_push` to `ruppell_pop`.
- **The engine** gives each kernel the innermost aten op open on its thread at
  its launch point: the launching call's start for an eager kernel; for a
  replayed kernel, the creation of its captured node (CUDA) or its launch
  while capturing (ROCm).
- **Exports**: the Chrome trace shows ops and scopes as nested host slices and
  each kernel's op in its args; `--format torch` writes the real ops (`cpu_op`,
  with `Input Dims` and `Input type`) and scopes (`nn.Module:` events), eager
  launches inside their real ops, replayed kernels under their captured op.

## Checked on B200 (2026-09-30)

- PyTorch demo (`examples/torch_graph_demo.py`): all 2,641 kernels have an
  op, the 2,400 replayed ones through capture (`model.1.fc2` → `aten::addmm`
  → an `nvjet` GEMM and a cuBLAS split-K reduction); the Chrome trace loads in
  Perfetto with no error stats; TraceLens categorizes the ops (GEMM,
  NORM_fwd, elementwise) and computes FLOPs, bytes and TFLOPS/s per shape.
- Qwen-AgentWorld-35B-A3B at TP2 under vLLM (the C4 configuration, `vllm
  bench serve`, 1,044 engine steps): 6.2 M aten ops and 16.5 k signatures per
  rank; 99.0% of kernels (98.2–99.0% of GPU time) and 99.5% of replayed
  kernels have an op.
- Offline: `tests/test_offline_cuda.py` (kernels to ops through eager calls
  and capture, the torch export's real ops).

Not yet run on ROCm: `tool.cpp` has the same hooks and the library the same
source, but no gfx950 job has shown them.

## Cost

The MoE at TP2, concurrency 8, 1,024 tokens in and 128 out, three arms
alternating per round, medians of 3 rounds against the untraced server
(`tests/cuda/run_cost.sh`, build `6cbec9d`, the same code as this branch
before its rebase onto main):

| Arm | Output throughput | Mean TPOT | Mean TTFT |
| --- | --- | --- | --- |
| traced | −4.3% | +2.9% | +14.9% |
| traced + CPU op tree (with inputs) | **−9.1%** | **+6.6%** | **+33.4%** |

The op tree roughly doubles the tracer's cost. Decode steps replay graphs
and run few aten ops on the CPU; prefill and the eager breaks run thousands
per step, each now with a callback and its inputs captured. `=names` (no
input capture) is not measured yet. It stays off by default; turn it on for
the runs whose question needs it.

## What it shows on the MoE (rank 0, engine steps 500–520, TraceLens)

| | |
| --- | --- |
| Categories | elementwise 35.5%, other (vLLM custom ops) 35.6%, GEMM 22.9%, reduce 4.5%, attention 1.4% |
| Top ops | `aten::mm` 22.9%, `vllm::moe_forward_shared` 21.9%, `aten::copy_` 16.6%, `aten::mul` 7.0%, `aten::add` 5.3%, `vllm::all_reduce` 5.2%, GDN attention core 4.7%, `aten::mean` 4.5% |

1. **The unfused RMS norm, op by op.** `aten::pow`, `aten::mean` and
   `aten::rsqrt` run 1,701 times each (81 norms × 21 steps), with the
   norm's casts: `(8, 2048)` activations bf16 → fp32 → bf16 (3,381 each way)
   and its `(2048,)` weight cast to fp32 on every call (1,701 times). The
   casts alone take 22.7 ms, 14% of the rank's kernel time; the weight cast
   (6.9 ms) could be done once.
2. **Decode GEMMs are small and far from bandwidth.** All `aten::mm` in the
   window have M = 8; memory-bound (~8 FLOPs/byte), they reach 1.2 TB/s
   time-weighted. The 8×**1**×2048 matrix-vector products (the shared
   expert's gate) take 6.4 ms at 0.01 TB/s, each a separate launch of a few
   µs; N = 32 and N = 256 are alike: candidates for fusing into a
   neighbouring kernel.
3. **2% of kernel time has no op** (TraceLens rows named after the launch
   call, `cudaLaunchKernel`, `cuLaunchKernelEx`): kernels launched outside
   any aten op; which code launched them is not yet traced.
