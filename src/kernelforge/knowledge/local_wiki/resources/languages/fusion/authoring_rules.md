---
title: Fusion authoring rules — CUDA-graph safety and the seven non-negotiables
kind: skill
scope: languages/fusion
updated: 2026-08-14
---

# Fusion authoring rules

Every rule below was paid for by a failure on real hardware. A fusion that
violates rule 4 passes a standalone microbench and then crashes the sglang
scheduler decode loop.

## The seven non-negotiables

1. **Env-gated.** With the flag unset the code path is bit-for-bit the original
   eager path. This is what makes the change safe to ship as a patch against an
   installed framework tree.
2. **fp32 accumulation inside the kernel.** Cast bf16 to fp32 in-kernel, not
   outside it. Accumulating outside defeats the point and changes numerics.
3. **One launch replaces the chain.** Fewer launches is the entire win. A
   "fusion" that still issues three kernels has not earned anything.
4. **CUDA-graph safe.** See below — this is the rule that bites.
5. **Import the real eager op as the parity oracle.** Never re-implement the
   reference; you will reproduce your own bug in both arms. Keep every public
   signature and import intact.
6. **ROCm-native Triton only.** Never reuse a framework CUDA-only fused op.
   `fused_qk_norm_rope` pulls in `cuda_bf16.h` and nvcc-only `--use_fast_math`;
   it will not build on ROCm.
7. **Fall back to eager if Triton is unavailable.** Never crash.

## CUDA-graph safety (rule 4, expanded)

Your kernel runs inside the captured decode CUDA graph. Capture happens once and
is replayed across varying batch sizes, so anything resolved at capture time is
frozen.

- **Static launch grid.** Never size the grid from a runtime or host value.
- **Preallocate every scratch and output tensor once, outside the fused path.**
  No per-call `torch.empty` / `zeros` / `cat`.
- **No host control flow on device data.** Never read `.item()` or a dynamic
  `.shape` into a Python branch.
- **No host<->device sync** anywhere in the decode hot path.
- **Index strictly in bounds for every token count.** One capture serves many
  batch sizes; an index that is only valid for the capture-time batch will read
  out of bounds on replay.
- Use `tl.constexpr` for shapes.

The symptom when this is wrong is not a wrong number. It is `HSA_STATUS_ERROR`,
a hardware exception, a memory access fault, or the scheduler taking SIGQUIT —
usually only under load, and never in the microbench.

## Numerics

bf16 with fp32 accumulation is not bit-exact against an eager path that
accumulates in a different order. Gate on SNR:

    snr_db = 10 * log10(sum(ref^2) / sum((ref - fused)^2))

with a >= 30 dB threshold. Do not use strict `allclose`. If you cannot reach the
gate, the fusion is wrong — widening the tolerance hides a real defect.

## Hybrid and Mamba models

`bench_one_batch` cannot initialize the Mamba/SSM backend on ROCm, so for hybrid
models the decode microbench is simply unavailable. Gate on kernel parity alone
and report the microbench as skipped. A skipped microbench is not a failure and
must not be scored as one.

## Triton limits on gfx942

Keep BLOCK size and shared-memory usage bounded and `tl.constexpr` shapes fixed,
or the kernel will not JIT-compile. "out of resource" and "shared memory" in a
Triton compile error both point here.

## Proven fusions

All four were authored and validated on the real sglang serving path with CUDA
graph on.

- **ZAYA CCA QK post-processing** (`ZAYA_FUSED_QK`): ~15-20 tiny fp32
  view/mean/add/mul/pow/sum/rsqrt ops into one Triton kernel, one program per
  (token, k-head). +14.7% e2e alone.
- **ZAYA ResidualScaling** (`ZAYA_FUSED_RESIDUAL`): dual affine `(x+bias)*scale`
  on the hidden and residual streams in one launch, bf16->fp32 in-kernel.
  Together with the QK fusion above: +34.5% e2e.
- **LFM2** (`LFM2_FUSED_RESIDUAL` / `LFM2_FUSED_SILU`): per-layer residual adds
  threaded into the next RMSNorm; w1|w3 SwiGLU merged into one GEMM plus a fused
  SiluAndMul. About +16% e2e.
- **Granite** (`GRANITE_FUSED_RESIDUAL`): `scaled_add_rmsnorm` folding scalar-mul,
  residual-add and RMSNorm into one kernel; ~5e-9 against eager.
