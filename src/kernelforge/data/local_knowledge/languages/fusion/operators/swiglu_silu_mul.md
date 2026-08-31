---
title: swiglu_silu_mul — Merge the gate/up SwiGLU projections into one GEMM and use the fused SiluAndMul kernel.
kind: operator
scope: languages/fusion
updated: 2026-08-14
---

# swiglu_silu_mul

Merge the gate/up SwiGLU projections into one GEMM and use the fused SiluAndMul kernel.

| | |
|---|---|
| Env flag | `FUSED_SILU` |
| Trigger categories | `activation`, `mul` |
| Minimum trigger share | 0.03 |
| Frameworks | sglang, vllm, vllm-aiter |

## What to fuse

Replace two separate gate/up projections + eager `F.silu(gate) * up` with a single MergedColumnParallelLinear([intermediate]*2) GEMM followed by the framework's fused `SiluAndMul` activation. Update weight loading to map gate->shard0, up->shard1.

## How to localize it in the source

Grep the model file for these anchors and fuse the chain they mark:

- `F.silu(`
- `silu(gate) * up`
- `self.w1(`
- `self.w3(`
- `gate_up_proj`
- `SiluAndMul`

## Correctness reference

Reference = eager `F.silu(gate) * up` on the same inputs. For the merged-GEMM part, compare against the two original Linear ops; import the framework SiluAndMul for the fused activation. Do NOT re-implement silu.

## Already-fused markers

If the source already matches one of these, the chain is fused and there is
nothing to claim here:

- `SiluAndMul`
- `gate_up_proj`

## ROCm constraint

Author a ROCm-native Triton (or aiter) kernel. Do not reuse a framework
CUDA-only fused op such as `fused_qk_norm_rope`. Verify the kernel builds and
runs on the target GPU, not only that it matches numerically.
