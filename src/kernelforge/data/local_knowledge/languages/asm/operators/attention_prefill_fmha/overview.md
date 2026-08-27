---
title: attention_prefill_fmha — overview
kind: operator_overview
operator: attention_prefill_fmha
gens: [gfx90a, gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz]
regimes: [prefill]
updated: 2026-06-08
sources:
  - https://rocm.blogs.amd.com/software-tools-optimization/ck-tile-flash/README.html
  - https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/workload.html
---

# attention_prefill_fmha  (FlashAttention forward, prefill / paged-KV)

## TL;DR
Fused multi-head attention for the prefill phase (`softmax(QKᵀ·scale + mask)·V`, online-softmax, no
materialized scores). On MI300X the **default and generally fastest** backend is **CK-Tile FMHA**;
Triton FA is the editable alternative; TileLang ≈1.5× Triton (CDNA3); asm is peak. **FlashInfer is
NVIDIA-only — N/A on AMD.**

## Math contract
`O = softmax( (Q·Kᵀ)·scale + mask )·V`, computed tile-wise with running max/sum (FA-2 style). Inputs
Q[b,h,sq,d], K/V paged; head_dim ≤ 256 supported on the CK ROCm backend (fwd+bwd). Causal/sliding-window
/ alibi are mask variants. Q/K head dim K0=128, V/O head dim N1=128 in the reference shape.

## Shape regimes
Prefill: long sq=sk (e.g. 4096), batch×heads folded into the grid. Distinct from
attention_decode_paged (sq=1, paged KV, latency-bound).

## Where it matters (Amdahl)
On hybrid/dense LLMs at ISL=1024 the full-attention prefill kernel is a *small* slice (~1%) when only
some layers are full-attn (e.g. Qwen3.5: 16/64 layers) — but the `--attention-backend` choice still
gave a real +~4–5% e2e (scheduling/graph side effects), and it converts the layer to an **editable**
Triton kernel surface. On attention-heavy models the slice is much larger.

## Backend landscape (→ SOTA cards)
| backend | status | card |
|---|---|---|
| ck (ck_tile) | 🟢 sota (default FA) | ck |
| aiter | 🟢 (MHA/MLA tuned) | backends/aiter.md (P2) |
| fa_rocm | 🟢 (CK+Triton backends) | backends/fa_rocm.md (P2) |
| triton | 🟡 (editable; default flag swap target) | backends/triton.md (P2) |
| tilelang | 🟡 (~1.53× Triton, CDNA3) | backends/tilelang.md (P2) |
| asm | 🟢 (peak; hard) | backends/asm.md (P2) |
| flashinfer | ⚪ na (NVIDIA-only) | — |

## Fusion neighbors
qk-norm / rope pre-step fusion; fp8 QKV; output proj. See [fusion.md](fusion.md) (P2).

## Numerics
Online-softmax fp32 accumulation; cross-backend bf16 argmax flips on long greedy decode are benign
(numerical-equivalence-class), not regressions — gate with parity probe ≥10 prompts.

## How to bench
Reference shape b=64, sq=sk=4096, d=128; isolated FMHA bench + oracle, or e2e via `--attention-backend`.

## Sources
- CK-Tile FA-v2 implementation + default backend: https://rocm.blogs.amd.com/software-tools-optimization/ck-tile-flash/README.html
- `VLLM_USE_TRITON_FLASH_ATTN=0` selects CK; head_dim≤256: ROCm workload guide + flash-attention ROCm.
- FlashInfer NVIDIA-only (SGLang AMD fallback): research 2026-06.
