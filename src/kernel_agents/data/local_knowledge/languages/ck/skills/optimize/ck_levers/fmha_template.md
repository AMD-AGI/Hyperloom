---
title: CK-Tile FMHA template — FlashAttention-2 forward/backward on Instinct
kind: language
gens: [gfx90a, gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz]
regimes: [prefill, decode, training]
status: sota
updated: 2026-06-08
sources:
  - https://rocm.blogs.amd.com/software-tools-optimization/ck-tile-flash/README.html
  - https://github.com/ROCm/composable_kernel/tree/develop/example/ck_tile/01_fmha
---

# CK-Tile FMHA template

## TL;DR
CK-Tile FMHA (`example/ck_tile/01_fmha`, kernels `fmha_fwd` / `fmha_bwd`, incl. **paged-KV**) is the
**production FlashAttention-2** on Instinct — the backend behind flash-attention ROCm and selectable in
vLLM/sglang. The forward kernel maps FA-2 one-to-one onto CK-Tile tiles. This file is the template/knob
deep-dive; for the SOTA decision card see
`operators/attention_prefill_fmha/backends/ck.md`, and for the general tile model
[ck_tile.md](ck_tile.md).

## Core concepts — FA-2 → CK-Tile mapping
| FA-2 step | CK-Tile mechanism |
|---|---|
| S = Q·Kᵀ | `gemm0` (a BlockGemm pipeline) → S tile in registers |
| m = rowmax(S) | `block_tile_reduce` (max) across the distribution |
| P = exp(S − m) | `sweep_tile` lambda over the per-lane Y elements |
| ℓ = rowsum(P) + correction | `block_tile_reduce` (sum) + running-stat rescale |
| O = P·V (+ rescale prev O) | `gemm1` BlockGemm; O accumulator rescaled by `exp(m_prev − m)` |

Online softmax accumulates in **fp32**. The kernel is assembled (like GEMM) from
`TilePartitioner + FmhaPipeline + EpiloguePipeline`; `generate.py` instantiates it per trait
(see [codegen_instances.md](codegen_instances.md)).

## The levers
### Pipeline variants (swap into `fmha_fwd_kernel`)
| Pipeline | dataflow | best for |
|---|---|---|
| `qr_ks_vs` | Q in **r**egisters, K/V streamed via **s**mem | general prefill |
| `qr_ks_vs_async` | + `buffer_load` async K/V direct-to-LDS | latency-hidden prefill (MI300X default) |
| paged-KV variants | KV gathered through a block/page table | **decode** with paged KV-cache (sglang/vLLM) |

The `qr` pipeline family also handles arbitrary head-dim padding.

### Knobs that matter on MI300X
- **Head-dim tile** `kK0`/`kK1` = 64/128 (fwd supports head_dim ≤256).
- **`kM0`** (Q rows per block) = 64/128 — the main occupancy/reuse lever.
- **`qr_ks_vs_async` vs sync** — async (DGL) is the latency-hidden default.
- **Causal/sliding/alibi masking specialization** — a separate codegen trait; the masked variant skips
  upper-triangle tiles.
- **Page size** for paged-KV (decode).
- **WarpGemm** for gemm0/gemm1: 32×32×8 (bf16) or 16×16×16; fp8 KV-cache uses the fp8 WarpGemm + per-tile
  scale (16×16×32 / 32×32×16).
- **Bias/rotary** fused via `bias.hpp` / `rotary.hpp` traits.

### Build & run
```bash
sh ../script/cmake-ck-dev.sh ../ gfx942
ninja tile_example_fmha_fwd
./bin/tile_example_fmha_fwd -b=1 -h=8 -s=4096 -d=128 -v=1   # validate vs reference
```

## Pitfalls
- **Backward pass tuning is separate and heavier** than forward — `fmha_bwd` has its own pipelines and
  trait set; don't assume forward tuning carries over.
- `generate.py` trait explosion — prune head-dims/dtypes/masks to your serving shapes or the FMHA codegen
  emits hundreds of `.cpp` files.
- Classic-CK `DeviceBatchedGemmSoftmaxGemm*` is **legacy** — do not use it for new attention work; CK-Tile
  FMHA superseded it.
- fp8 FMHA needs correctly-encoded (fnuz on CDNA3) scaled inputs; mismatched scale → silent garbage.

## Verify
- `-v 1` runs the example's built-in reference comparison.
- For a server integration: greedy temp=0 fixed-seed parity vs a reference attention (≥10 prompts);
  isolated FMHA bench vs the Triton backend at the same shape; confirm the backend banner in the log
  (`VLLM_USE_TRITON_FLASH_ATTN=0` / `--attention-backend ck`).

## Sources
- From Theory to Kernel: FlashAttention-v2 with CK-Tile (ROCm Blog — fmha pipeline mapping, qr_ks_vs, softmax→gemm1): https://rocm.blogs.amd.com/software-tools-optimization/ck-tile-flash/README.html
- ck_tile 01_fmha example (files, `generate.py`, `fmha_fwd_kernel.hpp`, `FmhaPipeline`/`EpiloguePipeline`, paged-KV): https://github.com/ROCm/composable_kernel/tree/develop/example/ck_tile/01_fmha
