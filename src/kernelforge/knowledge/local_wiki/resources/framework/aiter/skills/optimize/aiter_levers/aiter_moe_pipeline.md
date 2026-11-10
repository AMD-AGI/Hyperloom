---
title: aiter fused-MoE pipeline — what fuses, what the DB keys on, what misses
kind: lever
backend: aiter
gens: [gfx942, gfx950, gfx1250]
dtypes: [bf16, fp8_e4m3_fnuz, int8, fp4_e2m1]
regimes: [prefill, decode]
status: sota
updated: 2026-08-28
sources:
  - ROCm/aiter@b467ce342:aiter/fused_moe.py
  - ROCm/aiter@b467ce342:aiter/ops/moe_sorting.py
  - ROCm/aiter@b467ce342:aiter/configs/tuned_fmoe.csv
  - ROCm/aiter@b467ce342:csrc/ck_gemm_moe_2stages_codegen/gemm_moe_tune.py
  - https://rocm.blogs.amd.com/software-tools-optimization/wide-ep-deepseek/README.html
---

# aiter fused-MoE pipeline

## Route here when
- A MoE model is the workload and you need to know **what the single lever is** (it is the DB, not a
  kernel flag).
- You deployed a tuned `tuned_fmoe.csv` and saw no change.
- A MoE shape raises `device_gemm does not support this GEMM problem`.
- You are choosing a quantization for a MoE model and want to know which one unlocks the fast path.

**Skip this if** the model is dense — MoE tuning shares the capture→tune→deploy mechanics with dense
GEMM but nothing else. Go to [tuning_db.md](../../../overall/tuning_db.md).

## The shape of the thing
`aiter.fused_moe` is one call that swallows four stages:

```
token sorting  →  grouped GEMM stage 1 (gate + up)  →  activation  →  grouped GEMM stage 2 (down)
                                                                      + weighted combine
```

Everything past the entry point is chosen for you: the **quantization** selects the kernel family, and
the **per-shape DB** (`tuned_fmoe.csv`) selects the specific stage-1 and stage-2 kernels. There is no
"MoE block size" argument you tune by hand. AMD reports up to **3×** over an unfused stack.

The practical consequence: *your* lever is the DB and the quant choice. Everything else is a lookup.

## Why the two stages look different
A real shipped DB row names both kernels, and they are not from the same world:

```
stage 1 (fp8):  _ZN5aiter48fmoe_stage1_bf16_pertokenFp8_g1u1_64x128_2tg_pf3E
stage 2 (fp8):  moe_ck2stages_gemm2_256x64x128x256_1x4_MulABScaleExpertWeight_v3
                _Nswizzle0_Quant2_MulRoutedWeight1_F8_F8_B16
```

Stage 1 is typically a **hand-written asm** kernel; stage 2 is a **CK 2-stage** kernel. Reading the
name tells you what fused:
- `g1u1` — gate and up are fused into one GEMM
- `MulRoutedWeight1` — the router weight multiply landed in stage 2's epilogue
- `MulABScaleExpertWeight` — A/B scales and expert weight folded into the same epilogue

This asymmetry matters when you debug: an asm stage-1 failure and a CK stage-2 failure look nothing
alike. CK stage 2 is where coverage gaps live.

## Sorting is itself dispatched
`moe_sorting` produces `sorted_token_ids` / `sorted_expert_ids` and a padded block layout so the
grouped GEMM sees contiguous per-expert tiles. Padding is
`topk_ids.numel() + num_experts * block_size - topk`.

Sorting is **no longer a single kernel**. `fused_moe.py` picks among CK, Opus
(`moe_sorting_opus_fwd`), FlyDSL (`_flydsl_moe_sorting`), and an adaptive path
(`_adaptive_moe_sort`), by shape and quant. If a profile shows unexpected time in sorting, that is a
real dispatch decision, not fixed overhead.

## Quant routing — the biggest lever
| `quant_type` | Path |
|---|---|
| `QuantType.No` (bf16) | bf16 asm fused MoE |
| `per_Token` / `per_Tensor` fp8 (E4M3FNUZ) or int8 | block / per-token scaled CK + asm |
| A4W4 (FP4 weights) | FlyDSL when available, else **CK** ([aiter_flydsl_libtype.md](aiter_flydsl_libtype.md)) |

Note the enum crosses the custom-op boundary as its **value**, not the enum object — a torch custom-op
schema restriction. `fused_moe_` is registered as a custom op with a `fused_moe_fake` meta
implementation so the whole pipeline survives `torch.compile`.

Weight shapes: `w1` is `[num_experts, 2*inter, hidden]` (gate+up concatenated), `w2` is
`[num_experts, hidden, inter]` (down).

## The `tuned_fmoe` DB
Shipped header (**no `gfx` column**):

```
cu_num, token, model_dim, inter_dim, expert, topk, act_type, dtype,
q_dtype_a, q_dtype_w, q_type, use_g1u1, doweight_stage1, block_m, ksplit   # key
us1, kernelName1, err1, us2, kernelName2, err2, us, run_1stage, tflops, bw, _tag   # result
```

**The runtime lookup keys on `(gfx, cu_num, token, …)` — `gfx` is prepended**, backfilled from
`cu_num` for legacy CSVs. So a DB tuned on another arch misses, exactly like dense GEMM.

`token` is not the raw token count: it is the M-bucket from `get_padded_M`, which uses tier logic
(`_PADDED_M_TIERS`) — **not** the older "≤16 → 16, else nextPow2" rule. If you are reasoning about
whether your live shape will hit a tuned row, look at the tiers, not at pow2.

Deploy with `AITER_CONFIG_FMOE=/abs/tuned_fmoe.csv` (`:`-mergeable, see
[config_files_and_merge.md](../../../overall/config_files_and_merge.md)).

**A trap for anyone reusing the GEMM workflow:** the MoE tuner's `--errRatio` default is **0.5**, not
the 0.05 of the GEMM tuners. MoE stage tolerances are deliberately looser. If you copy a GEMM tuning
command and paste `--errRatio 0.05` in, you will discard most candidates.

```bash
# capture, then tune
python csrc/ck_gemm_moe_2stages_codegen/gemm_moe_tune.py \
    -i aiter/configs/untuned_fmoe.csv -o /tmp/tuned_fmoe.csv     # errRatio defaults to 0.5
```

A separate **grouped-MoE** DB exists for gfx1250 FlyDSL: `AITER_CONFIG_GROUPED_FMOE` →
`tuned_grouped_fmoe.csv`, with a much wider tile-config schema. SGLang's block-MoE path is gated by
`SGLANG_ROCM_AITER_BLOCK_MOE=1` and `CK_BLOCK_GEMM=1`.

## Shared-expert fusion (DeepSeek)
DeepSeek-style models run a shared expert for *every* token. AMD added a flag-gated path
(`fused_moe_dp_shared_expert` family) that folds that shared MLP into the FusedMoE kernel, removing a
separate Linear plus residual add while preserving the math. It was co-designed with MoRI-EP for
distributed DeepSeek — see `framework/mori/`.

This is worth checking specifically because it is a *structural* win (one fewer kernel per layer per
token), not a tuning win, so no amount of DB sweeping will find it.

## Verify
| Check | Command / signal | Pass condition |
|---|---|---|
| aiter MoE engaged at all | `AITER_LOG_MORE=1` | `fmoe_stage1_*` and `moe_ck2stages_*` appear, not a Triton MoE kernel |
| DB row hit | `AITER_LOG_TUNED_CONFIG=1` | `is tuned on cu_num` lines for MoE shapes; count > 0 |
| The win is real | tok/s on the MoE model, before vs after deploying the CSV | per `common_methodology/profiling/measure_protocol.md` |
| Accuracy held | a task metric, not `allclose` | fp8/A4W4 changes need an end-to-end gate |

## Failure modes
| Symptom | Cause | Fix |
|---|---|---|
| `device_gemm does not support this GEMM problem` | CK stage-2 instance gap on an odd expert/inter shape | pad to a covered shape, or tune so a covered instance is selected |
| Tuned CSV changes nothing | key miss — wrong `gfx`, `cu_num`, or quant signature | capture live, never hand-author `untuned_fmoe.csv` |
| Tuning discards nearly all candidates | `--errRatio 0.05` copied from the GEMM workflow | drop it; the MoE default 0.5 is correct |
| A4W4 slow or unsupported | FlyDSL absent → CK fallback | [aiter_flydsl_libtype.md](aiter_flydsl_libtype.md) |
| Profile shows surprising sorting cost | sorting backend dispatched to a slower path for this shape | it is a real decision — check which of CK/Opus/FlyDSL/adaptive fired |

## Numerics
Block and per-token fp8 introduce quantization error; DB rows carry `err1`/`err2` per stage (stage-2
values around 2.3% are normal). The fusion is designed to preserve the unfused math, so a fusion
change is parity-safe in principle — but a **quant** change is not. Gate on end-to-end task accuracy,
not on kernel tolerance.

## Deeper
[tuning_db.md](../../../overall/tuning_db.md) (the capture→tune→deploy discipline) ·
[config_files_and_merge.md](../../../overall/config_files_and_merge.md) (CSV resolution and merge) ·
[dispatch_and_rebind.md](../../../overall/dispatch_and_rebind.md) (how a call reaches aiter at all) ·
[operator_catalog.md](../../../overall/operator_catalog.md) (entry points and signatures) ·
[aiter_flydsl_libtype.md](aiter_flydsl_libtype.md) (A4W4) · `framework/mori/` (the EP dispatch/combine seam).

## Sources
- On-box `ROCm/aiter@b467ce342`: `aiter/fused_moe.py` (entry, custom op + fake impl, quant routing,
  gfx-first runtime key, `get_padded_M` tiers, sorting-backend dispatch, shared-expert path),
  `aiter/ops/moe_sorting.py` (padding formula, block layout),
  `aiter/configs/{tuned_fmoe,untuned_fmoe,tuned_grouped_fmoe}.csv` (schemas and real kernel names),
  `csrc/ck_gemm_moe_2stages_codegen/gemm_moe_tune.py` (`errRatio` default 0.5),
  `aiter/ops/flydsl/moe_kernels.py` (A4W4 stages),
  `aiter/jit/core.py` (`AITER_CONFIG_FMOE` / `AITER_CONFIG_GROUPED_FMOE` resolution and merge).
- Shared-expert fusion + MoRI-EP co-design (DeepSeek):
  https://rocm.blogs.amd.com/software-tools-optimization/wide-ep-deepseek/README.html
- Up to 3× fused MoE (AMD-reported, MI300X):
  https://rocm.blogs.amd.com/software-tools-optimization/aiter-ai-tensor-engine/README.html
