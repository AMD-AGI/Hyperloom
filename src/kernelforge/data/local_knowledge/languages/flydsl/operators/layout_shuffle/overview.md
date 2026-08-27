---
title: layout_shuffle — overview
kind: operator_overview
operator: layout_shuffle
gens: [gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz, fp8_e4m3, fp4_e2m1, int8]
regimes: [prefill, decode, both]
updated: 2026-06-08
---

# layout_shuffle  (weight pre-shuffle / bpreshuffle)

## TL;DR
A **one-time, offline** permutation of a weight tensor into an **MFMA-friendly layout** so the GEMM kernel
loads operand fragments with no in-kernel reshuffle and conflict-free LDS access. aiter's `shuffle_weight`
produces exactly this; the result carries `is_shuffled=True`, which flips the **`bpreshuffle` field of the
dense-GEMM 9-tuple dispatch key** ([[operators/dense_gemm/backends/aiter.md]]) so the GEMM dispatches to a
bpreshuffle kernel. It is the "pay the transpose once at load, not per GEMM" idea — the static-weight analog
of the [[operators/paged_kv_copy/overview.md]] shuffled-KV trick.

## Math contract
- **value-preserving permutation**: `W_shuf = permute(W, MFMA_layout)` — same elements, reordered so the
  matrix core reads its operand tiles directly. dtype unchanged.
- aiter `shuffle_weight(x, layout=(IN,IK), use_int4=False)` reshapes `[..., N, K]` into
  `[..., N/IN, K/BK, BK/Kel, IN, Kel]` and permutes to interleave `(IN, Kel)` last — i.e. groups the
  N-lane × K-pack the MFMA fragment expects. `BK = 2·IK`, `Kel = 16/elt_size` (32 for int4). Common layouts:
  `(16,16)` (default), `(32,16)` (a8w8 asm). Asserts `N % IN == 0`, `K % BK == 0`.
- variants: `shuffle_weight_NK(inst_N, inst_K)` (MFMA-instruction-shaped), `shuffle_weight_a16w4` /
  `shuffle_scale_a16w4` (FP4 MoE weight + its block scale).
- The **inverse** is implied by the consuming kernel — a shuffled weight is only valid for the kernel that
  expects that layout.

## Shape regimes
- **dense Linear weights** `[N,K]` (qkv/o/gate-up/down): shuffled once at model load.
- **MoE expert weights** `[E, N, K]` / FP4 `[E, N, K/2]`: per-expert shuffle (`shuffle_weight_a16w4`).
- One-time cost amortized over **every** forward pass → effectively free at steady state.

## Where it matters (Amdahl)
Not a runtime kernel on the hot path — it runs **once at load**. Its impact is **indirect but large**: it
unlocks the **bpreshuffle GEMM kernels** (asm/CK/FlyDSL) that are faster than the non-shuffled path, and it is
a required input to the FP4/FP8 fast MoE kernels. The GEMM speedup it enables is the real Amdahl lever
([[operators/dense_gemm/overview.md]]); the shuffle itself just must be correct and match the kernel's expected
layout.

## Backend landscape
Competitive positioning for this operator; this folder's card is [flydsl.md](flydsl.md). Other backends
live in their own folders (`framework/aiter/`, `languages/{triton,ck,asm,hip}/`).

| backend | status |
|---|---|
| aiter | 🟢 sota (`shuffle_weight`, ties to `bpreshuffle` key + tuned bpreshuffle CSVs) |
| hip | 🟢 (custom layout permute when authoring a bespoke MFMA kernel) |
| triton | 🟡 competitive (offline permute in torch/Triton; usually torch.permute suffices) |
| flydsl | 🟡 competitive (bpreshuffle B layout `(16*pack_n,16)` + MXFP scale layout + pow2 index lowering) |

## Fusion neighbors
Inseparable from **dense GEMM** (the bpreshuffle dispatch) and **fused MoE** (FP4/FP8 expert weights). Also
fuse with **weight quantization** at load: shuffle + quantize the weight in the same offline pass. See
[fusion.md](fusion.md) and, for the dense-GEMM bpreshuffle dispatch, [../dense_gemm/overview.md](../dense_gemm/overview.md).

## Numerics
Value-preserving permutation → byte-exact (per element); the GEMM that consumes it carries the usual
quant/accumulation numerics. The only correctness risk is a **layout mismatch** (wrong `layout=` vs the
kernel). See [numerics.md](numerics.md).

## How to bench
Don't bench the shuffle in isolation (one-time). Bench the **GEMM with vs without** the shuffled weight +
`bpreshuffle` kernel on the target shape; oracle = GEMM output `allclose` to the unshuffled-weight GEMM.
