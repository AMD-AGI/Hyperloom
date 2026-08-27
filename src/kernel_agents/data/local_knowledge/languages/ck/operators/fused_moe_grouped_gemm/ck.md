---
title: fused_moe_grouped_gemm on Composable Kernel — SOTA card
kind: sota_card
operator: fused_moe_grouped_gemm
backend: ck
gens: [gfx942, gfx950]
dtypes: [bf16, fp8_e4m3_fnuz, fp8_e4m3, int8, fp4_e2m1, mxfp4]
regimes: [prefill, decode]
status: sota
updated: 2026-06-08
sources:
  - https://github.com/ROCm/composable_kernel
  - ROCm/aiter@a6bb499375849eec45d68c5ccaebc8865fd422c0:aiter/configs/tuned_fmoe.csv
  - https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/optimizing-with-composable-kernel.html
  - https://github.com/ROCm/aiter/issues/2946
---

# fused_moe_grouped_gemm × CK

## TL;DR
> CK is the **stage-2 (down-proj) grouped-GEMM** behind aiter's fused MoE — the `moe_ck2stages_*` kernels,
> with block-scale fp8/int8 and the routed-weight/quant epilogue **fused into the GEMM**. It is the robust,
> broad instance library: you *consume* it (confirm shape coverage + let the DB pick the instance), you
> rarely author it by hand for serving. CK also supplies the block-scale GEMM path SGLang/aiter reach via
> `CK_BLOCK_GEMM=1`. The risk is **instance coverage** — a missing tile for an odd expert/inter shape is a
> crash, not a fallback.

## SOTA implementation(s)
| impl | source | gens/dtypes | measured perf | when best |
|---|---|---|---|---|
| `moe_ck2stages_*` (stage-2 grouped GEMM) | CK instances via aiter `tuned_fmoe.csv` | gfx942/950; fp8 block-scale, int8, bf16 | part of the up-to-3× fused MoE (AMD-reported, aiter blog) | the down-proj stage of fused MoE |
| CK block-scale GEMM | `ROCm/composable_kernel` (rocm-libraries) | gfx942 fp8 FNUZ; gfx950 OCP/fp4/mxfp4 | — | `CK_BLOCK_GEMM=1` block-scale MoE |
| CK-Tile grouped GEMM (authoring) | `include/ck_tile/ops/gemm`, `include/ck_tile/ops/fused_moe` | gfx942/950 | — | new MoE grouped-GEMM authoring |

### Decoding a real stage-2 instance name (from on-box `tuned_fmoe.csv`)
```
moe_ck2stages_gemm2_256x64x128x256_1x4_MulABScaleExpertWeight_v3_Nswizzle0_Quant2_MulRoutedWeight1_F8_F8_B16
                    └─ tile M×N×K×? ─┘ └wave┘ └── epilogue: A·B scale + expert weight ──┘  └pipe v3┘        └ in/out dtype ┘
```
- `256x64x128x256` — block tile (M 256, N 64, K 128, …).
- `MulABScaleExpertWeight` + `Quant2` — fp8 A-scale × B-scale × expert(routed) weight folded into the
  epilogue; `MulRoutedWeight1` = routed weight multiplied **in stage-2** (vs `doweight_stage1` in asm
  stage-1). `Nswizzle0` — no N-swizzle. `F8_F8_B16` — fp8 in, bf16 out.

The same `ck_tile/ops/fused_moe` tree also provides the `moe_sorting` kernel CK uses for the align&sort
layout (see moe_routing_topk).

### Measured stage-1+stage-2 timings (on-box `tuned_fmoe.csv`, 1700 rows)
The DB records the **winning** stage-1 (`us1`) and stage-2 (`us2`) per shape, measured on the build's SKU.
Schema: `cu_num,token,model_dim,inter_dim,expert,topk,act_type,dtype,q_dtype_a,q_dtype_w,q_type,use_g1u1,
doweight_stage1,block_m,ksplit,us1,kernelName1,err1,us2,kernelName2,err2,us,...,tflops,bw`.

| config (cu_num/token/model/inter/E/topk/quant) | stage-1 us1 | stage-2 us2 | total us | tflops | bw GB/s | source |
|---|---|---|---|---|---|---|
| 80 / 512 / 6144 / 4096 / E8 / top2 / fp8 per-Token | 373.4 | 268.5 (`moe_ck2stages_..._F8_F8_B16`, err 2.3%) | 641.9 | 240.9 | 955.6 | `tuned_fmoe.csv` |
| 80 / 512 / 6144 / 4096 / E8 / top2 / int8 per-Tensor | 386.1 | 250.0 (`...Quant1...I8_I8_B16`, err 2.1%) | 636.1 | 243.1 | 964.3 | `tuned_fmoe.csv` |
| 80 / 4 / 2304 / 1536 / E8 / top2 / fp8 per-Token (decode) | 17.7 | 15.1 (`256x32x64x256_..._F8_F8_B16`, err 0.3%) | 32.8 | 5.2 | 2591.4 | `tuned_fmoe.csv` |

(These are the DB's recorded per-kernel times on the build SKU — `cu_num=80` rows; treat as build-specific, not
portable. The decode row is bandwidth-bound, ~2.6 TB/s; the prefill rows are FLOP-bound, ~240 TFLOPs ≈ 45% of
MI300X fp8 peak, consistent with the ~45% achieved-vs-peak reality check.) The `err2` column is the stage-2
fp8/int8 kernel tolerance gated end-to-end (see numerics).

## Config space / knobs
| knob | where | note |
|---|---|---|
| instance tile (M/N/K block) | CK instance | e.g. 256×64×128×256; `ckProfiler` sweeps, aiter DB records the winner |
| MFMA instr | instance | 16×16 / 32×32 |
| pipeline version (`v3`) | instance | software-pipeline depth |
| `Nswizzle` | instance | N-dim reorder for L2 reuse |
| vector widths | instance | global load/store width |
| `MulRoutedWeight{0,1}` | instance/epilogue | where routed weight lands |
| `Quant{0,1,2}` | instance/epilogue | none / A-scale / A·B-scale |
| `IsSupportedArgument` | runtime gate | instance must cover (M,N,K, strides, dtype, layout) or it's **skipped** |
| `CK_BLOCK_GEMM=1`, `SGLANG_ROCM_AITER_BLOCK_MOE=1` | env | route to CK block-scale path |
| `CK_USE_FP8_ON_UNSUPPORTED_ARCH` | env | do **NOT** set on gfx942 (it has native fp8) |

## Numerics / parity
Same-math grouped GEMM, different tiling → parity-safe in bf16; fp8/fp4 block-scale adds quant error gated by
the recipe (the DB `err`). fp32 acc. FNUZ fp8 (native MFMA, `fmed3f` clip) on gfx942; OCP fp8 / fp4 / mxfp4
on gfx950. EP keeps the down-proj reduction local; TP shards N and must reduce across ranks in bf16/fp32 — do
not accumulate fp8 partials cross-rank. See [numerics.md](numerics.md).

## Integration (rebind seam)
You don't call CK's C++ by hand for serving — aiter wraps the `moe_ck2stages_*` instances and dispatches by
the DB. Ensure the **packaged CK build** has instances for your expert/inter shapes; `ckProfiler` confirms.
`CK_BLOCK_GEMM=1` / `SGLANG_ROCM_AITER_BLOCK_MOE=1` route to the CK block-scale stage-2. To author a new
instance, add it under `ck_tile/ops/fused_moe`, rebuild CK, regenerate the aiter DB entry.

## Pitfalls & anti-patterns
- ⚠ **Missing instance coverage** for odd expert/inter shapes → "device_gemm does not support" crash (not a
  fallback). aiter #2946: SIGABRT / GPU memory fault for E≥128, H=7168, ntok≥64 in `fused_moe_2stages` on
  gfx942. Pad inter to a covered multiple, rebuild with the dtype, or generate an instance.
- CK header-only transition (upcoming ROCm): `ckProfiler`/static libs may not be packaged — build standalone
  if your flow depends on them.
- A4W4/fp4 CK fast kernels need **CDNA4** HW (gfx950); on gfx942 they fall back / are absent.
- Setting `CK_USE_FP8_ON_UNSUPPORTED_ARCH` on gfx942 forces an emulation path though the HW has native fp8 —
  slower and unnecessary.

## How to verify
`AITER_LOG_MORE=1` → `moe_ck2stages_*` fired (not a Triton fallback); `ckProfiler` to confirm/select the
instance for your shape; grouped-GEMM output vs torch reference (bf16) or the DB `err` (fp8).

## Worked example (stage-2 coverage check before deploy)
DeepSeek-V3: E=256, inter 2048, hidden 7168, fp8 per-1x128.
1. `AITER_LOG_MORE=1` on a warmup batch → look for a `moe_ck2stages_gemm2_*_F8_F8_B16` line.
2. If you instead see "device_gemm does not support", run `ckProfiler` for `(M=block, N=7168, K=2048, fp8)`
   to find a covering instance; if none, pad inter or enable `SGLANG_MOE_PADDING` and use Triton stage-2.
3. Gate fp8 with a small eval; confirm the routed-weight multiply point (`MulRoutedWeight1`) matches your
   `doweight_stage1` setting so the weight isn't applied twice.

## Alternatives / cross-links
fused_moe_grouped_gemm · aiter (driver) · hip (stage-1 asm) ·
triton (fallback) · `backends/composable_kernel_lib/api.md` ·
`languages/composable_kernel/ck_tile.md` ·
[overview.md](overview.md) · [numerics.md](numerics.md).

## Sources
- CK as library / instances / `IsSupportedArgument`: https://github.com/ROCm/composable_kernel ; https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/optimizing-with-composable-kernel.html
- stage-2 kernel name: `ROCm/aiter@a6bb49937:aiter/configs/tuned_fmoe.csv`.
- stage-2 coverage crash (E≥128, H=7168): https://github.com/ROCm/aiter/issues/2946
