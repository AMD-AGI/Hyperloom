---
title: aiter fused MoE — fused_moe, shared-expert fusion, block-scale, tuned_fmoe DB
kind: backend
backend: aiter
gens: [gfx942, gfx950, gfx1250]
dtypes: [bf16, fp8_e4m3_fnuz, int8, fp4_e2m1]
regimes: [prefill, decode]
status: sota
updated: 2026-07-14
sources:
  - ROCm/aiter@b467ce3425cceeafe4f5587212d36df46feeb265:aiter/fused_moe.py
  - ROCm/aiter@b467ce3425cceeafe4f5587212d36df46feeb265:aiter/configs/tuned_fmoe.csv
  - ROCm/aiter@b467ce3425cceeafe4f5587212d36df46feeb265:csrc/ck_gemm_moe_2stages_codegen/gemm_moe_tune.py
  - https://rocm.blogs.amd.com/software-tools-optimization/wide-ep-deepseek/README.html
---

# aiter fused MoE (fmoe)

## TL;DR
`aiter.fused_moe` is **the MoE primitive on AMD serving stacks** — it fuses token sorting → grouped GEMM
(stage-1 gate+up, stage-2 down) → activation → weighted combine into one pipeline, auto-selecting the
kernel by **quant method** (bf16 asm / fp8 block-scale / int8 / A4W4 FlyDSL→CK). Like dense GEMM it is
**DB-driven**: `aiter/configs/tuned_fmoe.csv` maps a per-shape/per-quant key to the winning stage-1/stage-2
kernels. AMD reports up to **3×** vs an unfused stack; for DeepSeek, a flag-gated **shared-expert fusion**
folds the shared MLP into the same kernel.

## Concepts

### Entry + custom-op wrapper
`fused_moe(hidden_states, w1, w2, topk_weights, topk_ids, quant_type=QuantType.No, ...)` →
`fused_moe_` (registered as a torch custom op with a `fused_moe_fake` meta impl so it survives
torch.compile). `quant_type` is passed as the enum **value** across the custom-op boundary (schema
restriction) and converted back to `QuantType` inside. `w1` is `[num_experts, 2*inter, hidden]`
(gate+up), `w2` is `[num_experts, hidden, inter]` (down).

### Token sorting
`moe_sorting` (`aiter/ops/moe_sorting.py`) computes `sorted_token_ids / sorted_expert_ids` and a padded
block layout (`block_size`, default `BLOCK_SIZE_M`) so the grouped GEMM sees contiguous per-expert tiles.
Padding is `topk_ids.numel() + num_experts*block_size - topk`. Sorting is now itself dispatched across
backends — CK, **Opus** (`moe_sorting_opus_fwd`), **FlyDSL** (`_flydsl_moe_sorting`), and an adaptive path
(`_adaptive_moe_sort`) in `fused_moe.py` — chosen by shape/quant.

### Two-stage grouped GEMM
The DB (`tuned_fmoe.csv`) records, per shape, the chosen **stage-1** and **stage-2** kernels and a
`block_m`/`ksplit`. Real shipped rows show the kernel naming:
- stage-1 fp8: `_ZN5aiter48fmoe_stage1_bf16_pertokenFp8_g1u1_64x128_2tg_pf3E`
- stage-2 fp8: `moe_ck2stages_gemm2_256x64x128x256_1x4_MulABScaleExpertWeight_v3_Nswizzle0_Quant2_MulRoutedWeight1_F8_F8_B16`

i.e. stage-1 is often a hand-tuned asm kernel and stage-2 a **CK 2-stage** kernel (`moe_ck2stages_*`).
`g1u1` = gate-and-up fused; `MulRoutedWeight1` folds the router weight into the epilogue.

### Quant routing
`fused_moe` inspects `quant_type` / `q_dtype_a` / `q_dtype_w`:
- `QuantType.No` bf16 → bf16 asm fused MoE.
- `per_Token` / `per_Tensor` fp8 (E4M3FNUZ) or int8 → block/per-token scaled CK+asm path.
- A4W4 (FP4) → FlyDSL when available, else **CK** ([flydsl_path.md](flydsl_path.md)).

### Shared-expert fusion (DeepSeek)
DeepSeek-style models have a shared expert run for every token. AMD added a flag-gated path
(`fused_moe_dp_shared_expert` family) that **fuses the shared-expert MLP into the FusedMoE kernel**,
eliminating the separate Linear + residual add while preserving numerics. Co-designed with MoRI-EP
(expert-parallel comms) for distributed DeepSeek.

## The tuned_fmoe DB
Shipped `tuned_fmoe.csv` header (no `gfx` column):
`cu_num, token, model_dim, inter_dim, expert, topk, act_type, dtype, q_dtype_a, q_dtype_w, q_type,
use_g1u1, doweight_stage1, block_m, ksplit` → results `us1, kernelName1, err1, us2, kernelName2, err2, us,
run_1stage, tflops, bw, _tag`. **The runtime lookup in `fused_moe.py` keys on `(gfx, cu_num, token, …)`** —
`gfx` is prepended (legacy CSVs backfilled from `cu_num`), so a DB tuned on another arch misses. `token` is
the M-bucket from `get_padded_M` (tier logic `_PADDED_M_TIERS`, not the old ≤16→16/nextPow2 rule).
Deploy via `AITER_CONFIG_FMOE=/abs/tuned_fmoe.csv` (`:`-mergeable — see [configs_db.md](../../../overall/configs_db.md)).
Capture untuned shapes into `untuned_fmoe.csv` and tune with the MoE 2-stage tuner
`csrc/ck_gemm_moe_2stages_codegen/gemm_moe_tune.py` — its **`--errRatio` default is 0.5** (not the 0.05 of
the GEMM tuners; MoE stage tolerances are looser). A separate **grouped-MoE** DB exists for gfx1250 FlyDSL:
`AITER_CONFIG_GROUPED_FMOE` → `tuned_grouped_fmoe.csv` (wide tile-config schema). SGLang block-MoE path:
`SGLANG_ROCM_AITER_BLOCK_MOE=1`, `CK_BLOCK_GEMM=1`.

## The levers
- **Quant choice** (`quant_type`, `q_dtype_a/w`) — biggest perf knob; fp8 block-scale and A4W4 unlock the
  fast kernels.
- **`block_m` / `ksplit`** — grouped-GEMM tile + K-split, tuned per shape in the DB.
- **`use_g1u1`** (fuse gate+up), **`doweight_stage1`** (where the routed weight multiply lands).
- **shared-expert fusion flag** for DeepSeek.

## Numerics / parity
Block/per-token fp8 introduces quant error; DB rows carry `err1`/`err2` (e.g. stage-2 ~2.3%). The fusion
is designed to preserve the math of the unfused stack; validate end-to-end accuracy, not just kernel
tolerance, when changing quant.

## Pitfalls
- CK stage-2 coverage gaps → "device_gemm does not support this GEMM problem" for odd expert/inter shapes;
  pad or tune to a covered shape.
- A4W4 fallback to CK when FlyDSL missing (see [flydsl_path.md](flydsl_path.md)).
- DB key includes `gfx`, `cu_num`, and `token` (the M-bucket) + full quant signature — a capture/live
  mismatch (wrong arch / CU count / quant) misses, same failure mode as dense GEMM.

## How to verify
`AITER_LOG_MORE=1` to confirm `fmoe_stage1_*` / `moe_ck2stages_*` kernels fire (not a Triton MoE
fallback); compare tok/s on a MoE model before/after deploying a tuned `tuned_fmoe.csv`.

## Alternatives / cross-links
[flydsl_path.md](flydsl_path.md) (A4W4) · [configs_db.md](../../../overall/configs_db.md) · [integration.md](../../../overall/dispatch_and_rebind.md)
· operators: `fused_moe_grouped_gemm`, `shared_expert_fusion`, `moe_dispatch_combine`.

## Sources
- On-box: `ROCm/aiter@b467ce342`: `aiter/fused_moe.py` (entry, custom-op, quant routing, gfx-first key,
  `get_padded_M` tiers, sorting-backend dispatch), `aiter/configs/tuned_fmoe.csv` + `untuned_fmoe.csv` +
  `tuned_grouped_fmoe.csv` (DB schemas, real kernel names), `csrc/ck_gemm_moe_2stages_codegen/gemm_moe_tune.py`
  (MoE tuner, `errRatio=0.5`), `aiter/ops/flydsl/moe_kernels.py` (A4W4 stages), `aiter/jit/core.py`
  (`AITER_CONFIG_FMOE` / `AITER_CONFIG_GROUPED_FMOE` merge).
- Shared-expert fusion + MoRI-EP co-design (DeepSeek): https://rocm.blogs.amd.com/software-tools-optimization/wide-ep-deepseek/README.html
- Up-to-3× fused MoE (AMD-reported, MI300X): https://rocm.blogs.amd.com/software-tools-optimization/aiter-ai-tensor-engine/README.html
