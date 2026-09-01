---
title: CK — the FMHA stack (FlashAttention-2, paged-KV)
kind: language
lever: ck_fmha_stack
gens: [gfx950]
updated: 2026-08-28
sources:
  - https://rocm.blogs.amd.com/software-tools-optimization/ck-tile-flash/README.html
  - https://github.com/ROCm/composable_kernel/tree/develop/example/ck_tile/01_fmha
---

# The FMHA stack

CK-Tile FMHA (`example/ck_tile/01_fmha`, kernels `fmha_fwd` / `fmha_bwd`, including **paged-KV**) is the
**production FlashAttention-2 on Instinct** — the backend behind flash-attention ROCm and selectable in
vLLM/sglang.

## Route here when
- Authoring or tuning attention in CK.
- Choosing an FMHA pipeline variant, or wiring up paged-KV decode.
- The FMHA codegen is emitting hundreds of files and you need to prune it.

**Do not use classic `DeviceBatchedGemmSoftmaxGemm*`** — it is legacy and superseded by this.

## FA-2 → CK-Tile, step by step

| FA-2 step | CK-Tile mechanism |
|---|---|
| `S = Q·Kᵀ` | `gemm0` (a BlockGemm pipeline) → S tile in registers |
| `m = rowmax(S)` | `block_tile_reduce` (max) across the distribution |
| `P = exp(S − m)` | `sweep_tile` lambda over the per-lane Y elements |
| `ℓ = rowsum(P)` + correction | `block_tile_reduce` (sum) + running-stat rescale |
| `O = P·V` (+ rescale prev O) | `gemm1` BlockGemm; O accumulator rescaled by `exp(m_prev − m)` |

**Online softmax accumulates in fp32** — this is a correctness requirement at long context, not an
optimization. The kernel is assembled like a GEMM:
`TilePartitioner + FmhaPipeline + EpiloguePipeline`, with `generate.py` instantiating it per trait
(`ck_instance_codegen.md`).

## Pipeline variants

Swap into `fmha_fwd_kernel`:

| Pipeline | Dataflow | Best for |
|---|---|---|
| `qr_ks_vs` | Q in **r**egisters, K/V streamed via **s**mem | general prefill |
| **`qr_ks_vs_async`** | + async K/V **direct-to-LDS** | **latency-hidden prefill — the default** |
| paged-KV variants | KV gathered through a block/page table | **decode** with paged KV-cache (sglang/vLLM) |

The `qr` family also handles arbitrary head-dim padding.

On gfx950 the async path benefits from the widened **128 b/lane** direct-to-LDS and
**read-with-transpose `ds`** loads — re-check that the emitted form is the 12/16-DWORD one.

## Knobs that matter

| Knob | Values | Note |
|---|---|---|
| **`kM0`** (Q rows per block) | 64 / 128 | the main occupancy/reuse lever |
| Head-dim tile `kK0` / `kK1` | 64 / 128 | forward supports head_dim ≤ 256 |
| `qr_ks_vs_async` vs sync | — | async is the latency-hidden default |
| Mask specialization | causal / sliding / alibi | a separate codegen trait; the masked variant **skips upper-triangle tiles** |
| Page size | — | paged-KV decode |
| WarpGemm for gemm0/gemm1 | bf16 or fp8 | fp8 KV-cache uses the fp8 WarpGemm + per-tile scale |
| Bias / rotary | `bias.hpp` / `rotary.hpp` | fused traits |

gfx950's **160 KiB LDS** (2.5× a 64 KiB part) directly relaxes the head-dim and `kM0` ceiling that used
to bind here — re-tune rather than inheriting a tile sized for 64 KiB.

## Build and run

```bash
sh ../script/cmake-ck-dev.sh ../ gfx950
ninja tile_example_fmha_fwd
./bin/tile_example_fmha_fwd -b=1 -h=8 -s=4096 -d=128 -v=1   # -v 1 validates vs reference
```

## Verify

| Check | How |
|---|---|
| Correctness | `-v 1` runs the example's built-in reference comparison |
| Server integration | greedy temp=0 fixed-seed parity vs a reference attention, ≥10 prompts |
| Perf | isolated FMHA bench vs the Triton backend at the same shape |
| **Engagement** | confirm the backend banner in the log (`VLLM_USE_TRITON_FLASH_ATTN=0` / `--attention-backend ck`) — a kernel that is not dispatched cannot be measured |

## Pitfalls

| Symptom | Cause | Fix |
|---|---|---|
| Backward pass slow after tuning forward | **`fmha_bwd` has its own pipelines and trait set** | tune it separately; forward tuning does not carry over |
| Codegen emits hundreds of `.cpp` files | uncapped `generate.py` trait product | prune head-dims / dtypes / masks to your serving shapes |
| Used classic softmax-GEMM | `DeviceBatchedGemmSoftmaxGemm*` is legacy | use CK-Tile FMHA |
| fp8 attention silently wrong | scale does not match the encoding — gfx950 is **OCP**, not FNUZ | re-cast, never bit-copy |
| Head-dim tile inherited from a 64 KiB part | LDS is 160 KiB here | re-tune `kM0` / head-dim tile |

Full list: `ck_traps.md`.

## Sources
- From Theory to Kernel: FlashAttention-v2 with CK-Tile (ROCm Blog — pipeline mapping, `qr_ks_vs`, softmax→gemm1): https://rocm.blogs.amd.com/software-tools-optimization/ck-tile-flash/README.html
- ck_tile 01_fmha example (files, `generate.py`, `fmha_fwd_kernel.hpp`, `FmhaPipeline`/`EpiloguePipeline`, paged-KV): https://github.com/ROCm/composable_kernel/tree/develop/example/ck_tile/01_fmha
