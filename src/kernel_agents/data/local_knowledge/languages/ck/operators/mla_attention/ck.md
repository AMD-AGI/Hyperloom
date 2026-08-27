---
title: mla_attention on CK — SOTA card
kind: sota_card
operator: mla_attention
backend: ck
gens: [gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz]
regimes: [prefill, decode]
status: competitive
updated: 2026-06-08
sources:
  - https://github.com/ROCm/composable_kernel/tree/develop/example/ck_tile/01_fmha
  - https://rocm.blogs.amd.com/software-tools-optimization/ck-tile-flash/README.html
  - ROCm/aiter@a6bb499375849eec45d68c5ccaebc8865fd422c0:aiter/mla.py
  - https://vllm.ai/blog/2026-02-27-rocm-attention-backend
---

# mla_attention × CK

## TL;DR (one-line decision)
> CK-Tile can express MLA (the `kv_lora_rank=512` latent + decoupled `qk_rope=64` attention) as an
> FMHA-family kernel, and CK is the templating layer AMD uses for new attention kernels. But on serving,
> MLA is routed to the **aiter asm `mla_decode_fwd`** (the tuned ceiling, 17× vs naive) — so CK MLA is
> "the from-source authoring option / the template under the asm path," not the typical serving choice.
> Use CK when building an MLA kernel from source or needing a CK-templated variant; otherwise use
> aiter.

## SOTA implementation(s)
| impl | source | gens / dtypes | measured perf | when it's best |
|---|---|---|---|---|
| CK-Tile MLA (FMHA-family, latent+rope) | `ROCm/composable_kernel:example/ck_tile/01_fmha` (+ MLA traits) | gfx942/950; bf16/fp16/fp8 | below aiter asm MLA on serving | from-source MLA authoring |

**Context — the asm path it competes with** (`aiter/mla.py`): aiter's `mla_decode_fwd` consumes the
absorbed latent cache `kv_buffer = [num_page, page_size, nhead_kv, kv_lora_rank + qk_rope_head_dim]`
(`512 + 64`) and runs `mla_decode_stage1_asm_fwd`. A CK MLA kernel must match this two-part-score
contract (latent GEMM + decoupled-RoPE GEMM, fused online softmax) to be a drop-in.

## Config space / knobs
| param | range / values | effect | default |
|---|---|---|---|
| latent-dim tile | 512 (`kv_lora_rank`) | LDS-heavy on gfx942 (64 KB) | 512 |
| rope-dim tile | 64 (`qk_rope_head_dim`) | decoupled-RoPE score part | 64 |
| WarpGemm | `16×16×16` (bf16); fp8 `16×16×32` | MFMA shape | 16×16×16 |
| page size | 1 / >1 | paged-KV gather | 1 |
| split-KV / persistent partitioner | derived | flash-decoding parallelism | auto |
| LDS budget | **64 KB MI300X, 160 KB MI350X** | the 512 latent tile fits easier on MI350X | gen-specific |

Codegen via FMHA `generate.py` (prune to the MLA head config). See
`languages/composable_kernel/fmha_template.md`.

## Numerics / parity
fp32 two-part score accumulate (latent + rope); matrix absorption exact in bf16; fp8 latent/KV via fp8
WarpGemm + scale — **FNUZ gfx942 / OCP gfx950** — accuracy gate. Oracle is aiter `mla_decode_fwd`. See
[numerics.md](numerics.md).

## Integration (rebind seam)
From source: build the CK-Tile MLA kernel; the `fmha_fwd`-family callable is the seam. CK does **not**
have a first-class serving MLA entry the way aiter does — most serving routes through aiter MLA. To wire a
CK MLA kernel you'd replace aiter's `mla_decode_fwd` import or ship it through aiter's catalog.
**Verify it engaged:** rocprofv3 → `fmha_*`/`*ck_*` (vs aiter `mla_decode_stage1_asm_fwd`).

## Pitfalls & anti-patterns
- aiter asm MLA is the tuned ceiling — don't pick CK MLA for serving without measuring against it.
- The **512-wide latent tile is LDS-heavy on gfx942 (64 KB)** — size tiles to fit; MI350X's 160 KB LDS
  gives more headroom.
- composable_kernel moved into `ROCm/rocm-libraries` (standalone deprecated) — pin the commit.
- `generate.py` trait explosion — prune to the MLA config.
- Must reproduce the absorbed-MQA contract (`nhead_kv=1`, latent 512 + rope 64) or parity breaks.

## Worked example
Authoring a CK MLA decode variant for a research change (e.g. different rope dim):
1. Start from `01_fmha` MLA traits; set latent tile 512, rope tile to your dim.
2. Size LDS to the gen (64 KB gfx942 / 160 KB gfx950) — the 512 latent tile is the binding constraint.
3. `-v 1` reference compare, then greedy temp=0 parity vs aiter `mla_decode_fwd`.
4. Bench vs aiter asm at the served ctx; ship only if it wins (it usually won't on standard MLA).

## How to verify (bench + oracle)
Build the CK MLA kernel, reference-compare; greedy temp=0 parity vs aiter `mla_decode_fwd`; isolated
decode bench vs aiter at the served shape; accuracy gate for fp8.

## Alternatives / cross-links
./aiter.md (serving SOTA) · ./triton.md (reference) · ../../attention_prefill_fmha/backends/ck.md ·
../../attention_decode_paged/backends/ck.md · `languages/composable_kernel/fmha_template.md` ·
`hardware/mi300x.md`, `hardware/mi350x.md` · ../overview.md.

## Sources
- CK-Tile FMHA family / MLA templating: https://github.com/ROCm/composable_kernel/tree/develop/example/ck_tile/01_fmha ; https://rocm.blogs.amd.com/software-tools-optimization/ck-tile-flash/README.html
- The asm contract it must match (latent 512 + rope 64, nhead_kv=1): on-box `ROCm/aiter@a6bb499375849eec45d68c5ccaebc8865fd422c0:aiter/mla.py`.
- MLA routed to aiter asm on serving: https://vllm.ai/blog/2026-02-27-rocm-attention-backend
