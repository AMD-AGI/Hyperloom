---
title: Decode-path kernel fusion — index, pattern cards and authoring rules
kind: index
scope: languages/fusion
updated: 2026-08-14
---

# Decode-path kernel fusion — knowledge map

Entry index for everything under `languages/fusion/`. Load this file whole; read
the individual cards on demand.

> **Convention (KernelForge standard):** a knowledge folder that contains an
> `INDEX.md` is navigated **through this file**. Folders without one fall back to
> a generated "filename — one-line description" listing.

## What this knowledge base is

How to collapse a chain of tiny decode-path operations in **sglang** or **vLLM**
into one env-gated Triton kernel on AMD Instinct, and how to prove the result
without crashing production serving.

This folder is about *where the launches go*, not about any single kernel being
slow. Once GEMM and attention are tuned, a decode step can still spend most of
its GPU-busy time in residual adds, RMSNorm, RoPE, activations and cache writes —
each paying a full launch. The win is arithmetic on launch count, not on FLOPs.

For the diagnosis that decides whether a workload is even a fusion candidate, see
`common_methodology/optimization/lever_fusion.md`. For Triton authoring
levers themselves (block sizes, `num_warps`, ISA verification) see
`languages/triton/`. This folder does not duplicate either.

## Reading order

1. **`authoring_rules.md`** — the seven non-negotiable rules and the CUDA-graph
   safety contract. Read this before writing any kernel; it is the difference
   between a fusion that ships and one that SIGQUIT-crashes the scheduler.
2. **`harness_contract.md`** — the validation harness you must emit, its JSON
   shape, and the warm-up trap that silently manufactures a 3% result.
3. **`operators/<pattern>.md`** — the specific chain you are fusing.

## Pattern cards

| Pattern | Chain | Env flag |
|---|---|---|
| `residual_add_rmsnorm` | residual add folded into the following RMSNorm | `FUSED_RESIDUAL` |
| `swiglu_silu_mul` | merged gate/up GEMM plus fused SiluAndMul | `FUSED_SILU` |
| `scaled_residual_add_rmsnorm` | `rmsnorm(branch*scale + residual)` | `GRANITE_FUSED_RESIDUAL` |
| `hybrid_scale_combine` | hybrid attn/mamba prescale and output combine | `FALCON_H1_FUSED_SCALES` |
| `qk_norm_rope` | per-head Q/K norm, optional blend and temperature, RoPE | `FUSED_QK` |
| `dual_affine_scaling` | `(x + bias) * scale` on hidden and residual streams | `FUSED_RESIDUAL_SCALE` |

Patterns are model-agnostic hypotheses. The chain they name has to be confirmed
against the actual framework source before it is worth authoring.

## Which framework file to edit

Fusion edits the serving framework's Python model definition, not `aiter/`. That
makes the change a patch against an installed tree, which is why every fusion is
env-gated: with the flag unset the file must behave exactly as it did before.

## Measured results

The four fusions in `authoring_rules.md` were validated end to end on real sglang
serving with CUDA graph enabled. They are the calibration for what a plausible
gain looks like: a single strong chain is worth a few percent, and stacking two
on the same model reached +34.5%.
