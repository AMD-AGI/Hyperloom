---
title: moe_dispatch_combine on Triton — SOTA card
kind: sota_card
operator: moe_dispatch_combine
backend: triton
gens: [gfx942, gfx950]
dtypes: [bf16, fp8_e4m3_fnuz]
regimes: [prefill, decode]
status: experimental
updated: 2026-07-29
sources:
  - ROCm/aiter@a177781d4:aiter/ops/triton/comms/all_gather.py
  - ROCm/aiter@a177781d4:aiter/ops/triton/comms/reduce_scatter.py
  - ROCm/aiter@a177781d4:aiter/ops/triton/comms/iris.py
  - https://github.com/ROCm/iris
  - https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/fused_moe/fused_moe.py
---

# moe_dispatch_combine × Triton

## TL;DR
> Triton handles the **local** permute/scatter (the gather/scatter inside a single-GPU fused-MoE Triton
> kernel) well, and there is an **experimental** GPU-initiated comms path: aiter ships Triton
> reduce-scatter / all-gather built on **Iris** (a SHMEM-like Triton library), confirmed present at
> `aiter/ops/triton/comms/{all_gather,reduce_scatter,iris}.py` on the current tree. For production EP
> dispatch/combine, use MoRI-EP (HIP) — Triton is the local-permute path and an emerging distributed-comm
> authoring surface.

## SOTA implementation(s)
| impl | source | gens/dtypes | measured perf | when best |
|---|---|---|---|---|
| Triton fused-MoE local permute/scatter | vLLM/sglang `fused_moe.py` Triton | gfx942/950 | — | single-GPU MoE permute (no EP) |
| aiter Triton comms (Iris-based) | `aiter/ops/triton/comms/{all_gather,reduce_scatter,iris}.py` | gfx942/950 | — (experimental) | authoring GPU-initiated comms in Triton (all-gather/reduce-scatter primitives; EP dispatch/combine itself is not shipped here) |

Recommend: Triton for local permute; **MoRI-EP (HIP)** for distributed dispatch/combine.

## Config space / knobs
- Local permute: `BLOCK` over tokens; wave64 `num_warps=2–4`; fp32 index math for scatter/gather offsets.
- Iris comms: GPU-initiated `iris.put`/`load`/`store` over a symmetric heap; persistent-grid PID mapping;
  `grid` sized to CU count.

## Numerics / parity
Same as the operator: fp8/fp4 dispatch quant gate, bf16 (or blockwise-quantized) combine, masked pad.
Triton local permute is a pure gather (lossless) — the numeric risk is downstream. See
[numerics.md](../numerics.md).

## Integration (rebind seam)
- vLLM/SGLang: the Triton fused-MoE path (vs. `VLLM_ROCM_USE_AITER=1`) uses the Triton permute.
- aiter Iris comms: `aiter/ops/triton/comms/` (requires the `iris` package).

## Pitfalls & anti-patterns
- Triton GPU-initiated comms are **experimental** — don't ship as the production EP dispatch/combine path;
  what's shipped today (`all_gather`/`reduce_scatter`) is not itself an EP all-to-all.
- `num_warps=8` from NVIDIA-tuned configs → spill on the permute kernel; use 2–4.
- Iris requires the symmetric heap + the `iris` library installed.

## How to verify
rocprof: confirm the Triton permute/comms kernel ran; round-trip identity + greedy parity; for Iris,
sanity-check bandwidth vs the MoRI-EP table (expect lower — it's experimental).

## Alternatives / cross-links
[mori.md](mori.md) (production EP) · [hip.md](hip.md) · [../aiter.md](../aiter.md) · [overview.md](../overview.md).

## Sources
- aiter Triton Iris comms (confirmed on current tree): `ROCm/aiter@a177781d4:aiter/ops/triton/comms/{all_gather,reduce_scatter,iris}.py`.
- Iris (Triton SHMEM): https://github.com/ROCm/iris
- Triton fused-MoE reference: https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/fused_moe/fused_moe.py
