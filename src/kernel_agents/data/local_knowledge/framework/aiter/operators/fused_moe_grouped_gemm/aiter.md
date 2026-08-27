---
title: fused_moe_grouped_gemm on aiter — SOTA card
kind: sota_card
operator: fused_moe_grouped_gemm
backend: aiter
gens: [gfx942, gfx950, gfx1250]
dtypes: [bf16, fp8_e4m3_fnuz, fp8_e4m3, int8, fp4_e2m1]
regimes: [prefill, decode]
status: sota
updated: 2026-07-14
sources:
  - ROCm/aiter@b467ce342:aiter/fused_moe.py
  - ROCm/aiter@b467ce342:aiter/configs/tuned_fmoe.csv
  - ROCm/aiter@b467ce342:aiter/ops/moe_sorting.py
  - https://rocm.blogs.amd.com/software-tools-optimization/aiter-ai-tensor-engine/README.html
  - https://www.amd.com/en/blogs/2025/revolutionizing-mixture-of-experts-performance-10.html
  - https://www.lmsys.org/blog/2026-05-28-mori/
  - https://rocm.blogs.amd.com/artificial-intelligence/mlperf-inference-v6.0/README.html
  - https://github.com/vllm-project/vllm/issues/34641
---

# fused_moe_grouped_gemm × aiter

## TL;DR
> `aiter.fused_moe` **is** the MoE grouped-GEMM mega-kernel on AMD serving stacks: `moe_sorting` (align&sort)
> → stage-1 (gate+up `g1u1`) → activation → stage-2 (down) → weighted combine, **DB-driven by quant
> method**. It is the live path on vLLM/SGLang; to improve it you tune `tuned_fmoe.csv`, not the underlying
> CK/asm. Stage-1 is usually a hand-tuned **asm** kernel (`fmoe_g1u1*`), stage-2 a **CK** instance
> (`moe_ck2stages_*`). fp8 block-scale unlocks **up to 3×** over an unfused stack (DeepSeek-V3, MI300X);
> the align&sort redesign cut the **sort step 10×** (AMD). The MoRI integration further fuses the cross-GPU
> EP dispatch/combine into the FusedMoE kernel (conceptual analog of NVIDIA-only DeepGEMM Mega-MoE), and
> MXFP4 GEMMs are ≈**62%** of Llama2-70B e2e cost — so the low-bit path is the lever. NOT for gfx942 FP4
> (CDNA4-only — falls back / crashes; see pitfalls).

## SOTA implementation(s)
| impl | source | gens/dtypes | measured perf | when best |
|---|---|---|---|---|
| `fused_moe` (DB-driven, asm stage-1 + CK stage-2) | `aiter/fused_moe.py` + `aiter/configs/tuned_fmoe.csv` | gfx942/950; bf16, fp8 block-scale (FNUZ/OCP), int8, A4W4 | up to **3×** vs unfused MoE on **DeepSeek-V3 @ MI300X** (AMD-reported, aiter blog); align&sort redesign **10×** on the sort step (AMD MoE blog) | the live MoE path on sglang/vLLM |
| `fused_moe_1stage` (asm single-kernel) | `aiter/fused_moe.py:367` (`fmoe_g1u1`, `fmoe_g1u1_tkw1`, `fmoe_fp8_blockscale_g1u1`, `fmoe_int8_g1u0`) | gfx942/950; fp8/int8/A4W4 per-Token/1x128/1x32 | part of the 3× path | per-shape asm winner from the DB |
| FlyDSL A4W4 stages | `aiter/ops/flydsl/moe_kernels.py` | gfx950 fp4 | — | A4W4 on CDNA4 (else CK fallback) |

The DB picks stage-1 (often **asm** `fmoe_g1u1*`) and stage-2 (**CK** `moe_ck2stages_*`) per (shape, quant).
This card subsumes the ck/hip/triton selection on the live path; consult ck and hip
for the two stages individually.

### What the code actually does (on-box `b467ce342`)
`fused_moe()` (`fused_moe.py:441`) is a torch custom op with `fused_moe_fake` (survives `torch.compile`). It
calls `fused_moe_()` → `fused_moe_1stage()`, which **dispatches on the quant signature** (the 1-stage asm
table is gfx-split: e.g. `per_1x128` → `fmoe_g1u1` on gfx942 vs `fmoe_fp8_blockscale_g1u1` on gfx950):

```python
# aiter/fused_moe.py — fused_moe_1stage(), the asm stage-1 selector
elif quant_type == QuantType.per_Token and doweight_stage1 and isG1U1:
    aiter.dynamic_per_token_scaled_quant(a8, hidden_states, a8_scale)
    aiter.fmoe_g1u1_tkw1(moe_buf, a8, w1, w2, sorted_ids, sorted_weights,
                         sorted_expert_ids, num_valid_ids, topk, a8_scale, ...)
...
if quant_type == QuantType.per_1x128:
    fmoe_func = functools.partial(aiter.fmoe_fp8_blockscale_g1u1,
                                  fc_scale_blkn=128, fc_scale_blkk=128, block_size_M=block_size_M)
elif isG1U1:
    fmoe_func = aiter.fmoe_g1u1                 # gate+up fused asm
else:
    aiter.fmoe_int8_g1u0(...)                   # int8, separate gate/up
```

A static `(activation, quant_type, dtype, q_dtype_a, q_dtype_w, isG1U1, doweight_stage1) → asm API` table
(the `fused_moe_1stage_dict`, gfx-split for gfx942 vs gfx950) routes every supported combination — e.g.
`(Silu, per_1x128, bf16, fp8, fp8, True, False) → fmoe_g1u1`, `(Silu, per_1x32, bf16, fp4x2, fp4x2, True,
False) → fmoe_g1u1` (MXFP4, gfx950). The `per_1x128` path quantizes the activation **inside** the asm kernel
(`xbf16` branch sets `a1 = hidden_states`, `a1_scale = empty`). Note `per_128x128` is remapped to
`per_1x128` on some paths (`moe_op.py`).

### Measured per-shape timings (on-box `tuned_fmoe.csv`, build SKU `cu_num=80`)
The DB stores the winning stage-1/stage-2 names **and** their measured times. Real rows:

| config (token/model/inter/E/topk/quant) | stage-1 us1 (asm) | stage-2 us2 (CK) | total us | tflops | bw GB/s |
|---|---|---|---|---|---|
| 512 / 6144 / 4096 / E8 / top2 / fp8 per-Token | 373.4 (`fmoe_stage1_bf16_pertokenFp8_g1u1_64x128_2tg_pf3`) | 268.5 (`moe_ck2stages_..._F8_F8_B16`) | 641.9 | 240.9 | 955.6 |
| 512 / 6144 / 4096 / E8 / top2 / int8 per-Tensor | 386.1 (`...pertokenInt8_g1u1_64x128_2tg_pf3`) | 250.0 (`...I8_I8_B16`) | 636.1 | 243.1 | 964.3 |
| 4 / 2304 / 1536 / E8 / top2 / fp8 per-Token (decode) | 17.7 (`...g1u1_32x64_4tg_pf3`) | 15.1 | 32.8 | 5.2 | 2591.4 |

Source: `ROCm/aiter@b467ce342:aiter/configs/tuned_fmoe.csv`. Build-specific (`cu_num=80`), not portable. Prefill
rows are FLOP-bound (~240 TFLOPs fp8 ≈ ~45% of MI300X peak — the achieved-vs-peak reality check); the decode row
is bandwidth-bound (~2.6 TB/s). `block_m=64` (prefill) / `32` (decode), `ksplit=0`.

## Config space / knobs
| knob | where | effect / range |
|---|---|---|
| `quant_type` / `q_dtype_a` / `q_dtype_w` | call arg | biggest lever: `No`/`per_Token`/`per_1x128`(block-scale)/`per_1x32`(MXFP4) × bf16/fp8/int8/fp4 |
| `block_size_M` (BLOCK_M) | DB + `moe_sorting` | grouped-GEMM tile M; **must match** the align&sort pad |
| `ksplit` | DB | stage-2 K-split for skinny down-proj |
| `use_g1u1` (`isG1U1`) | call arg | fuse gate+up into one stage-1 GEMM |
| `doweight_stage1` | call arg | apply routed weight in stage-1 (`fmoe_g1u1_tkw1`) vs in combine |
| `activation` | call arg | `ActivationType.Silu` / `Gelu` |
| shared-expert fusion | `fused_moe_dp_shared_expert` | DeepSeek shared expert folded into the dense path |
| `AITER_CONFIG_FMOE` | env | abs path to `tuned_fmoe.csv` (`:`-mergeable overlays) |
| `SGLANG_ROCM_AITER_BLOCK_MOE=1`, `CK_BLOCK_GEMM=1` | env | route to CK block-scale stage-2 |

The runtime DB key is `(gfx, cu_num, token-bucket M, full quant signature, E, inter_dim, model_dim)`. `gfx`
= `get_gfx()` and `cu_num` = `get_cu_num()` — capture on the **same** arch + SKU you serve on (e.g. gfx942
304 CU on MI300X) or every row misses. The token bucket comes from `get_padded_M` (tier logic, not the old
nextPow2 rule).

## Numerics / parity
fp32 accumulate. fp8 block-scale `err1`/`err2` in the DB are kernel tolerance (stage-2 ~2.3%) — these are
**per-kernel** and must be gated **end-to-end** (aiter #2421 reports fp8 MoE precision drift). FNUZ fp8 on
gfx942 (native MFMA); OCP fp8 / FP4 on gfx950. MXFP4 (`per_1x32`) uses a 32-element shared exponent; the
scale is re-sorted per token via `mxfp4_moe_sort_fwd` before stage-1. See [numerics.md](numerics.md).

### FP8 MoE vs EP-vs-TP placement
fp8 MoE error is dominated by the **stage-2 (down)** quant because activations after SwiGLU have a wide
dynamic range; keep stage-2 scales per-token or per-1x128, not per-tensor. EP (expert parallel) keeps the
full weight per expert local → quant is identical to single-GPU; TP (tensor parallel) shards the inter
dimension → the down-proj reduction crosses ranks and **must reduce in bf16/fp32**, never accumulate the
fp8 partials. This is why EP is the preferred sharding for fp8 MoE at scale.

## Integration (rebind seam)
`fused_moe(hidden, w1, w2, topk_weights, topk_ids, quant_type=..., doweight_stage1=...)` → custom op.
- vLLM: `VLLM_ROCM_USE_AITER=1` + `VLLM_ROCM_USE_AITER_MOE=1` (see
  `vllm/model_executor/layers/fused_moe/rocm_aiter_fused_moe.py`).
- SGLang: `SGLANG_USE_AITER=1` (+ `SGLANG_ROCM_AITER_BLOCK_MOE=1` for block-scale).
- Overlay a tuned CSV without editing site-packages: `AITER_CONFIG_FMOE=/abs/tuned_fmoe.csv`.

## Pitfalls & anti-patterns
- **Capture/live mismatch** (`cu_num`/token-bucket/quant signature) → tuned CSV inert, silent fall to a
  default config. Capture on the live SKU + shapes.
- **FP4 on gfx942**: A4W4/MXFP4 has no native HW on CDNA3 → falls back, and vLLM's
  `VLLM_ROCM_USE_AITER_FP4BMM` **defaults to True and crashes on MI300X** ("MXFP4 quantization is not
  supported on gfx942", vLLM #34641). Set `VLLM_ROCM_USE_AITER_FP4BMM=0` on gfx942.
- **CK stage-2 coverage gap** → "device_gemm does not support" (pad/tune the inter shape); aiter #2946
  reports SIGABRT / GPU memory fault for E≥128, H=7168, ntok≥64 in `fused_moe_2stages` on gfx942.
- **HIP-graph capture crash** (sglang #16025) → disable graph capture or force the Triton fused-MoE.
- `doweight_stage1` inconsistent with the reference weight-multiply point → wrong combine.

## How to verify
`AITER_LOG_MORE=1` → confirm `fmoe_g1u1*` (asm stage-1) + `moe_ck2stages_*` (CK stage-2) fire, not Triton.
Per-shape grouped-GEMM timing before/after the tuned CSV; e2e MoE tok/s A/B; greedy decode + a small eval
(MMLU subset) to gate fp8/fp4 quant. `op_tests/test_moe.py` for kernel parity vs torch.

## Worked example (DeepSeek-V3 MoE block, MI300X)
E=256, top-8, hidden 7168, inter 2048, fp8 per-1x128 block-scale, M=4096 tokens.
1. `biased_grouped_topk` → `topk_ids/topk_weights` (see moe_routing_topk).
2. `moe_sorting(topk_ids, block_size=block_size_M)` → `sorted_ids/sorted_expert_ids` padded to BLOCK_M.
3. `fused_moe(..., quant_type=per_1x128, isG1U1=True)` → `fmoe_fp8_blockscale_g1u1` stage-1 (quant inside),
   CK `moe_ck2stages_*` stage-2 with `MulRoutedWeight1`.
4. Verify: `AITER_LOG_MORE=1` shows both kernels; tok/s vs `VLLM_ROCM_USE_AITER_MOE=0` Triton baseline.
Expect the asm+CK path to beat Triton on this hot shape; if a stage-2 instance is missing, pad inter to a
covered multiple or fall to Triton.

## Alternatives / cross-links
fused_moe_grouped_gemm · ck (stage-2) · hip (stage-1 asm) ·
triton (portable fallback) · moe_routing_topk · moe_dispatch_combine ·
`backends/aiter/fmoe.md` · [overview.md](overview.md) ·
[numerics.md](numerics.md).

## Sources
- on-box: `ROCm/aiter@b467ce342:aiter/fused_moe.py` (`fused_moe:441`, `fused_moe_1stage`, gfx-split asm
  dispatch table, `get_padded_M` tiers, gfx-first key), `aiter/configs/tuned_fmoe.csv`,
  `aiter/ops/moe_sorting.py` (+ `moe_sorting_opus.py`), `aiter/ops/flydsl/moe_kernels.py`,
  `csrc/ck_gemm_moe_2stages_codegen/gemm_moe_tune.py` (MoE tuner, `errRatio=0.5`).
- up to 3× block-scale fused MoE (DeepSeek-V3, MI300X): https://rocm.blogs.amd.com/software-tools-optimization/aiter-ai-tensor-engine/README.html
- 10× align & sort (sort step): https://www.amd.com/en/blogs/2025/revolutionizing-mixture-of-experts-performance-10.html
- MoRI in-kernel EP dispatch/combine fusion + 2.56× A2A BW: https://www.lmsys.org/blog/2026-05-28-mori/
- MXFP4 GEMMs ≈62% of Llama2-70B e2e: https://rocm.blogs.amd.com/artificial-intelligence/mlperf-inference-v6.0/README.html
- fp8 MoE precision: https://github.com/ROCm/aiter/issues/2421
- stage-2 SIGABRT (E≥128, H=7168): https://github.com/ROCm/aiter/issues/2946
- FP4BMM default-on crash on gfx942: https://github.com/vllm-project/vllm/issues/34641
