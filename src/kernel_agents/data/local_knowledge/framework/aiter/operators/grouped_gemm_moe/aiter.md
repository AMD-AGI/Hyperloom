---
title: grouped_gemm_moe on aiter — SOTA card
kind: sota_card
operator: grouped_gemm_moe
backend: aiter
gens: [gfx942, gfx950, gfx1250]
dtypes: [bf16, fp16, fp8_e4m3_fnuz, fp4_e2m1]
regimes: [prefill, decode]
status: sota
updated: 2026-07-14
sources:
  - ROCm/aiter@b467ce342:aiter/fused_moe.py
  - ROCm/aiter@b467ce342:aiter/jit/core.py
  - https://github.com/ROCm/aiter
  - https://rocm.blogs.amd.com/artificial-intelligence/aiter-intergration-s/README.html
  - https://www.amd.com/en/blogs/2025/revolutionizing-mixture-of-experts-performance-10.html
  - https://www.lmsys.org/blog/2026-05-28-mori/
  - https://rocm.blogs.amd.com/artificial-intelligence/kimi-k2.5-optimize/README.html
  - https://github.com/ROCm/aiter/issues/915
---

# grouped_gemm_moe × aiter

## TL;DR
> On sglang/vllm, **aiter's `fused_moe` is the live MoE GEMM path** and the SOTA on AMD: a single fused
> asm/CK kernel does grouped gate/up → activation → grouped down → routing-weight combine, accelerating MoE
> **up to 3×** (DeepSeek-V3, MI300X) vs naive per-expert launches; the align&sort redesign cut the **sort
> step 10×** and **MoRI** fuses cross-GPU EP dispatch/combine into the kernel. Use it unless you hit a
> shape/dtype it doesn't cover (then fall back
> to ck/triton). Tuning is the same capture/deploy idea as dense GEMM, but via the **`tuned_fmoe.csv`**
> (`AITER_CONFIG_FMOE`) and a `(activation, quant_type, dtypes...)` dispatch table.

## SOTA implementation
`fused_moe` dispatches on a tuple of `(activation, quant_type, in/w/out dtype, doweight, ...)` to a concrete
fused kernel. From `aiter/fused_moe.py` (`ROCm/aiter@b467ce342`):

```python
def fused_moe(hidden_states, w1, w2, topk_weight, topk_ids, expert_mask=None,
              activation=ActivationType.Silu, quant_type=QuantType.No,
              doweight_stage1=False, w1_scale=None, w2_scale=None,
              a1_scale=None, a2_scale=None, block_size_M=None, splitk=0, ...):
    ...
# dispatch table (excerpt):
(ActivationType.Silu, QuantType.No,        bf16, bf16, bf16, False, False): aiter.fmoe,
(ActivationType.Gelu, QuantType.per_Token, bf16, fp8,  i4x2, True,  False): aiter.fmoe_g1u1,
# fp8 blockscale -> aiter.fmoe_fp8_blockscale_g1u1 ; int8 -> aiter.fmoe_int8_g1u0
```

`w1:[E, inter*2, dim]`, `w2:[E, dim, inter]`; per-expert/per-token scales `w1_scale/a1_scale/...` flow in
for quant paths. The grouped GEMM is "GEMM of different shapes" batched across experts in one launch — that
batching is where the ~3× comes from.

| impl | source | gens/dtypes | measured perf | when best |
|---|---|---|---|---|
| aiter asm `fused_moe` (grouped) | `aiter/fused_moe.py` → `aiter.fmoe*` | gfx942/950; bf16, fp8, A4W4 via FlyDSL | up to **3×** fused-MoE on DeepSeek-V3 (MI300X) via grouped-GEMM-of-different-shapes; align&sort **10×** on sort step (AMD) — no first-party isolated GEMM number reproduced | live MoE serving on sglang/vllm |
| aiter FlyDSL grouped (mixed precision) | FlyDSL backend (else CK) | gfx942/950; A4W4 mxfp | A4W4/MXFP4 **1.6× latency @ concurrency 512 (MI355X)**; Kimi-K2.5 −65% TTFT / −69% TPOT / +162% tput (vendor); auto-falls back to CK | low-bit MoE |

## Config space / knobs
| param | range / values | effect | default |
|---|---|---|---|
| `activation` | Silu / Gelu | fused activation between the two grouped GEMMs | Silu |
| `quant_type` | No / per_Token / per_Tensor / per_1x128 / per_128x128 / per_1x32 | selects quant kernel family (`per_128x128`→`per_1x128` on some paths) | No |
| `doweight_stage1` | bool | apply routing weight in stage-1 vs combine | False |
| `block_size_M` | -1 / tile | per-group M tile (tuning knob); -1 = kernel default | None→-1 |
| `splitk` | 0..n | K split for the grouped GEMM | 0 |
| `gate_mode` / `expert_mask` | int / tensor | routing mode / EP expert mask | — |
| `AITER_CONFIG_FMOE` | path(s) | deploy tuned MoE config CSV (own DB, NOT dense bf16) | `configs/tuned_fmoe.csv` |

The MoE DB key is the `tuned_fmoe` tuple `(gfx, cu_num, token, model_dim, inter_dim, expert, topk, quant
signature, …)` — **not** the dense-GEMM 10-tuple. Small per-group M is handled inside the grouped kernels,
not by routing to the dense skinny path.

## Numerics / parity
- fp32 accumulate; per-expert / per-block scales applied **after** the MFMA dot; **fp32 routing-weight
  combine** → [numerics.md](numerics.md). Quant MoE (fp8/A4W4) needs a **task-accuracy gate**, not
  byte parity.

## Integration (rebind seam)
Live call site: aiter `fused_moe` (invoked by the sglang/vllm MoE layer). Engages via the aiter env switches
the serving framework already sets; tune/deploy through `AITER_CONFIG_FMOE`. Verify with the tuned-config log
marker (see dense card) and by confirming the asm/CK fused-MoE kernel name in a rocprof trace.

## Pitfalls & anti-patterns
- **Coverage gap**: `fused_moe` can hit *"does not support this GEMM problem"* for some shapes/TP configs
  (e.g. Qwen3-235B-A22B BF16 TP8, aiter issue #915) → fall back to ck/triton for that shape.
- A4W4 path needs **FlyDSL installed**; missing FlyDSL silently uses CK (correct but slower for low-bit).
- The dispatch tuple must match exactly — wrong `activation`/`quant_type`/dtype combo raises a KeyError or
  routes to the wrong kernel; confirm against the table.
- Tuning the dense `bf16_tuned_gemm.csv` does **nothing** for MoE — MoE has its own `tuned_fmoe.csv`.

## How to verify (worked example)
```bash
rocprofv3 --stats -- <run one MoE forward>          # expect aiter.fmoe* / asm fused-MoE kernel
# A/B vs the triton/ck MoE path with the parity oracle in ../numerics.md;
# accept on non-overlapping latency improvement + accuracy gate held
grep -c 'is tuned' server.log                        # MoE tuned-config engagement
```

## Alternatives / cross-links
operators/grouped_gemm_moe/backends/triton · operators/grouped_gemm_moe/backends/ck ·
operators/dense_gemm/backends/aiter (same tune mechanism) ·
operators/dense_gemm/backends/flydsl (FlyDSL MoE backend) ·
operators/skinny_gemv_decode/backends/aiter (small per-group M) · quantization/block_scaling_mxfp ·
operators/grouped_gemm_moe/overview

## Sources
- On-box: `aiter/fused_moe.py` (`fused_moe:441`, signature + gfx-split dispatch table, gfx-first key),
  `aiter/jit/core.py` (`AITER_CONFIG_FMOE` / `AITER_CONFIG_GROUPED_FMOE`),
  `csrc/ck_gemm_moe_2stages_codegen/gemm_moe_tune.py` (MoE tuner) — `ROCm/aiter@b467ce342`.
- AITER repo & fused_moe: https://github.com/ROCm/aiter
- SGLang-AITER integration (up to 3× fused MoE, DeepSeek-V3, via grouped GEMM): https://rocm.blogs.amd.com/artificial-intelligence/aiter-intergration-s/README.html
- 10× align & sort (sort step): https://www.amd.com/en/blogs/2025/revolutionizing-mixture-of-experts-performance-10.html
- FlyDSL A4W4 1.6× @ concurrency 512 (MI355X) + MoRI in-kernel EP fusion: https://www.lmsys.org/blog/2026-05-28-mori/
- Kimi-K2.5 FlyDSL FusedMoE (−65% TTFT / −69% TPOT / +162% tput): https://rocm.blogs.amd.com/artificial-intelligence/kimi-k2.5-optimize/README.html
- Coverage-gap example: https://github.com/ROCm/aiter/issues/915
