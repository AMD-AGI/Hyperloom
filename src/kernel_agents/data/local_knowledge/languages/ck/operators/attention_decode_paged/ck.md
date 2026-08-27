---
title: attention_decode_paged on CK — SOTA card
kind: sota_card
operator: attention_decode_paged
backend: ck
gens: [gfx90a, gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz]
regimes: [decode]
status: competitive
updated: 2026-06-08
sources:
  - https://github.com/ROCm/composable_kernel/tree/develop/example/ck_tile/01_fmha
  - https://rocm.blogs.amd.com/software-tools-optimization/ck-tile-flash/README.html
  - ROCm/aiter@a6bb499375849eec45d68c5ccaebc8865fd422c0:aiter/ops/mha.py
  - https://vllm.ai/blog/2026-02-27-rocm-attention-backend
---

# attention_decode_paged × CK

## TL;DR (one-line decision)
> CK-Tile FMHA has **split-KV / paged-KV pipeline variants** (`-d fwd_splitkv`, `-d batch_prefill`) that
> handle decode (KV gathered through a block/page table), and CK is the backend under flash-attention
> ROCm. On AMD serving, however, decode is usually routed to the **aiter asm decode kernel**
> (`paged_attention_rocm`, partition 256) which is faster — so CK decode is "the templated kernel under
> the CK FA path / a from-source authoring option," not the typical serving SOTA. Prefer
> aiter; use CK when building a CK-Tile decode kernel from source or needing the CK FA
> path's paged decode.

## SOTA implementation(s)
| impl | source | gens / dtypes | measured perf | when it's best |
|---|---|---|---|---|
| CK-Tile FMHA split-KV / paged decode | `ROCm/composable_kernel:example/ck_tile/01_fmha` (`fwd_splitkv`, paged-KV) | gfx90a/942/950; bf16/fp16; head_dim ≤256 | the CK FA path's decode; generally below aiter asm decode | building from CK source; CK FA path |
| CK FA backend decode (via fa_rocm) | `ROCm/flash-attention` CK submodule | bf16/fp16, head_dim ≤256 | mature fp16/bf16, default `ck` backend ROCm ≥6.0 | stable half-precision decode |

**Real codegen seam** — aiter compiles the CK split-KV decode kernel via `generate.py`
(`aiter/ops/mha.py`):
```python
f"{CK_DIR}/example/ck_tile/01_fmha/generate.py -d fwd_splitkv "   # split-KV decode
f"{CK_DIR}/example/ck_tile/01_fmha/generate.py -d batch_prefill " # paged batch (chunked+decode)
```
Same `01_fmha` example as prefill — decode is the `kM0=1`-effective specialization with the split-KV
reduction and paged-KV gather traits enabled.

## Config space / knobs
| param | range / values | effect | default (decode) |
|---|---|---|---|
| `kM0` (Q rows) | 1 effective | decode = 1 query token | 1 |
| `kN0` (KV tile) | 64 / 128 | KV streaming granularity | 128 |
| `kK0`/`kK1` (head-dim tile) | 32 / 64 | head-dim chunking | 32 |
| WarpGemm | `16×16×16` / `32×32×8`; fp8 `16×16×32` | 16×16 preferred | 16×16×16 |
| page size | 16 / 32 / 64 | paged-KV gather granularity | 16 |
| split-KV count | derived from ctx | flash-decoding parallelism | auto |
| paged-KV gather variant | on/off | block-table indirection | on |

Codegen via the FMHA `generate.py` (prune to your decode head dims/dtypes). See
`languages/composable_kernel/fmha_template.md`.

## Numerics / parity
fp32 online-softmax accumulate; paged-KV gather doesn't change the math; split-KV reduce uses per-split
`m_i`. fp8 KV uses fp8 WarpGemm + scale — **FNUZ on gfx942, OCP on gfx950**. See
[numerics.md](numerics.md).

## Integration (rebind seam)
- Via flash-attention ROCm: `VLLM_USE_TRITON_FLASH_ATTN=0` → CK FA (includes paged/split-KV decode where
  built).
- From source: build the FMHA example with `fwd_splitkv` + paged-KV traits; the `fmha_fwd` callable is
  the seam.
- **Verify it engaged:** rocprofv3 → `fmha_*`/`*ck_*` (vs aiter `paged_attention_rocm` or Triton).

## Pitfalls & anti-patterns
- **head_dim ≤ 256** hard limit (CK FA).
- aiter asm decode generally beats CK decode on serving → don't pick CK decode without measuring against
  aiter.
- `generate.py` trait explosion — prune to your decode shapes.
- composable_kernel moved into `ROCm/rocm-libraries` (standalone deprecated) — pin the commit.
- Split-KV count too low at long ctx underutilizes CUs; too high adds reduce overhead.

## Worked example
Building a CK decode kernel for a head_dim=192 model (aiter custom path supports 192, but you want a CK
variant):
1. `generate.py -d fwd_splitkv` with `head_dim=192`, dtype bf16, paged-KV trait.
2. Build, `-v 1` reference compare vs PyTorch attention.
3. Bench isolated decode vs aiter `paged_attention_v2` at the served ctx; only ship CK if it wins.
4. rocprofv3 confirms `fmha_fwd_splitkv_*hd192*` fired.

## How to verify (bench + oracle)
Build the FMHA example with the split-KV/paged-KV decode trait, `-v 1` reference compare; isolated decode
bench vs aiter at the served shape; greedy temp=0 parity. Confirm CK fired (`*ck_*`/`fmha_*` in
rocprofv3) when selecting the CK FA path.

## Alternatives / cross-links
./aiter.md (faster decode) · ./triton.md · ../../attention_prefill_fmha/backends/ck.md ·
../../mla_attention/backends/ck.md · `languages/composable_kernel/fmha_template.md` · ../overview.md.

## Sources
- CK-Tile FMHA split-KV/paged decode codegen: on-box `ROCm/aiter@a6bb499375849eec45d68c5ccaebc8865fd422c0:aiter/ops/mha.py` (`-d fwd_splitkv`, `-d batch_prefill`); https://github.com/ROCm/composable_kernel/tree/develop/example/ck_tile/01_fmha ; https://rocm.blogs.amd.com/software-tools-optimization/ck-tile-flash/README.html
- aiter decode beats generic FA decode (route to aiter): https://vllm.ai/blog/2026-02-27-rocm-attention-backend
