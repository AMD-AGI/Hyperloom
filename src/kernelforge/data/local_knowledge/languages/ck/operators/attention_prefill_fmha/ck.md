---
title: attention_prefill_fmha on CK-Tile — SOTA card
kind: sota_card
operator: attention_prefill_fmha
backend: ck
gens: [gfx90a, gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz]
regimes: [prefill]
status: sota
updated: 2026-06-08
sources:
  - https://rocm.blogs.amd.com/software-tools-optimization/ck-tile-flash/README.html
  - ROCm/composable_kernel:example/ck_tile/01_fmha
  - ROCm/aiter@a6bb499375849eec45d68c5ccaebc8865fd422c0:aiter/ops/mha.py
  - https://github.com/Dao-AILab/flash-attention
---

# attention_prefill_fmha × CK-Tile

## TL;DR (one-line decision)
> CK-Tile FMHA is the **default, generally fastest** FlashAttention-2 forward on Instinct — it is the
> kernel behind `flash-attention` ROCm (the `ck` backend, default since FA-2 ROCm), and the kernel that
> aiter codegens and ships under its CK MHA path. Choose it as the **baseline** for prefill attention
> (bf16/fp16/fp8, head_dim ≤256, causal/SWA/alibi/paged-KV). Reach for Triton/TileLang/asm only when you
> need editability (Triton), the last 10–20% on a fixed shape (asm), or a feature CK lacks for your shape.

## SOTA implementation(s)
The CK-Tile FMHA example *is* the SOTA template; aiter and FlashAttention-ROCm both build from it.

| impl | source (`repo@commit:path`) | gens / dtypes / shapes | measured perf (`value @ hw, ROCm/lib, date`) | when it's best |
|---|---|---|---|---|
| CK-Tile FMHA fwd (FA-2) | `ROCm/composable_kernel:example/ck_tile/01_fmha` | gfx90a/942/950; bf16/fp16/fp8_fnuz; **head_dim ≤256 fwd+bwd**; causal/SWA/alibi/paged-KV | default FA backend on MI300X; vendor/community SOTA for general prefill (quote achieved TFLOPS per shape when measured on-box) | general prefill FMHA, paged-KV |
| CK FA backend via FlashAttention-ROCm | `Dao-AILab/flash-attention` `csrc/flash_attn_ck/` → CK submodule | MI200x/250x/300x/355x, RDNA3/4; head_dim ≤256 | default `ck` backend (vs Triton), ROCm ≥6.0, 2026 (vendor support matrix) | drop-in `flash_attn_func` on ROCm |
| aiter-shipped CK FMHA | `ROCm/aiter@a6bb49937:aiter/ops/mha.py` (codegen call below) | gfx942/950; bf16/fp16/fp8 | the CK path under `aiter.flash_attn_func`; part of the AITER FA **1.2–4.4× TPS** envelope (vendor) | production serving via aiter |

**Real codegen seam** — aiter compiles the CK FMHA example on the fly (`aiter/ops/mha.py`):
```python
# aiter/ops/mha.py (mha_fwd / batch_prefill compile_ops)
f"{CK_DIR}/example/ck_tile/01_fmha/generate.py -d fwd "          # plain prefill
f"{CK_DIR}/example/ck_tile/01_fmha/generate.py -d fwd_splitkv "  # split-KV (long ctx / decode)
f"{CK_DIR}/example/ck_tile/01_fmha/generate.py -d batch_prefill "# paged batch-prefill
```
`generate.py` emits instances per `(head_dim, dtype, causal, paged, alibi…)` — this is the "trait
explosion" you prune. `ENABLE_CK` (`aiter/jit/core.py`) gates whether aiter uses this CK path at all; if
unset, `flash_attn_func` falls through to the Triton kernel (see line 1984: `if not ENABLE_CK:`).

## Config space / knobs
CK FMHA tiling is a **template specialization**, not a runtime knob table — but these are the levers in
`generate.py` / the policy headers:

| param | range / values | effect | default (FMHA fwd) |
|---|---|---|---|
| `BlockSize` (workgroup) | 256 | warps per WG; drives occupancy | 256 (4 wave64) |
| `M0PerBlock` (Q rows/tile) | 64/128 | bigger = more reuse, more VGPR | 128 |
| `N0PerBlock` (KV cols/tile) | 64/128 | KV streaming granularity | 128 |
| `K0/K1PerBlock` (head-dim tile) | 32/64 (×n for d≤256) | head-dim chunking | 32 |
| WarpGemm (MFMA) | `16×16×16` / `32×32×8`; fp8 `16×16×32` | **16×16 > 32×32** (lower power, higher FLOPs) | 16×16×16 |
| `AK1`/`BK1` (vector load) | ≥8 elems (128-bit) | global-load width; want `global_load_dwordx4` | 128-bit |
| `kKPack` | %WarpSize==0 | LDS pack; **64 on MI300, 32 on RDNA3** | 8 |
| pipeline | `qr`/`async` (CDNA4 `q_waitcnt`) | overlap QK/softmax/PV | qr |

Aim ≥1024 workgroups; pick `mfma_16x16`; 128-bit loads. The `static_assert(kKPack % K3 == 0)` in
`block_fmha_bwd_pipeline_default_policy.hpp` is warp-size sensitive (64 on MI300, 32 on RDNA3).

## Numerics / parity
fp32 online-softmax accumulate (P/O stored bf16); identical math to FA-2. fp8 FMHA uses scaled
inputs (per-tensor descale, or KV-blockscale for paged). On gfx942 the fp8 dialect is **e4m3 FNUZ**;
on gfx950 it is **OCP e4m3** — feeding the wrong dialect mis-scales by ~2× (silent garbage). Cross-backend
bf16 tie-breaks (CK vs Triton vs asm have different reduction order) are benign — gate on greedy
temp=0 parity, not bit-exactness. See [numerics.md](numerics.md).

## Integration (rebind seam)
- **FlashAttention-ROCm:** the `ck` backend is the default; Triton is opt-in. `flash_attn_func` →
  `csrc/flash_attn_ck/mha_fwd.hip` → CK FMHA instance.
- **vLLM/sglang:** `VLLM_USE_TRITON_FLASH_ATTN=0` (or `--attention-backend ck`) selects the CK FA path.
- **aiter:** if `ENABLE_CK` is set, `aiter.flash_attn_func` uses the codegen'd CK kernel; the Python
  `flash_attn_func` / `mha_batch_prefill` callable is the capture/rebind seam for an authored
  replacement (attention has a clean Python forward seam — unlike a library GEMM).
- **Verify it engaged:** rocprofv3 Top-N should show `fmha_*` / `*ck_*` kernel names (not `_attn_fwd`
  Triton names); confirm the backend banner in the server log.

## Pitfalls & anti-patterns
- **head_dim ≤ 256 hard limit** — d>256 (some MLA-adjacent / custom heads) must use Triton.
- composable_kernel moved into `ROCm/rocm-libraries` (standalone repo deprecated) — pin the commit.
- `ckProfiler` may be absent in some images → can't sweep CK instances there; fall back to aiter/Triton.
- AITER CK can crash under HIP-graph capture for novel shapes (`device_gemm does not support this GEMM
  problem`, sglang #16025) → force Triton for that model/shape.
- Backward-pass tuning is separate and heavier; this card is the **forward** (prefill).
- `generate.py` trait explosion — prune `-d`/dtype/head-dim lists or build times balloon.

## Worked example
DeepSeek/Llama-style 128-head, head_dim=128, bf16, causal, ctx=4096 prefill:
1. `ENABLE_CK=1` so aiter uses CK; serve with `--attention-backend ck` (sglang) or
   `VLLM_USE_TRITON_FLASH_ATTN=0` (vLLM).
2. rocprofv3 → confirm `fmha_fwd_*_hd128_*` fires (not Triton `_attn_fwd_*`).
3. Bench isolated FMHA at `(B,H=128,sq=4096,sk=4096,d=128,causal,bf16)` vs the Triton backend; CK should
   win on plain bf16. If you flipped to fp8, check it picked the **FNUZ** path on gfx942.
4. Greedy temp=0 parity vs a reference attention over ≥10 prompts.

## How to verify (bench + oracle)
Greedy temp=0 fixed-seed parity vs a reference attention (≥10 prompts); isolated FMHA bench vs the Triton
backend at the same `(B,H,sq,sk,d,causal,dtype)`; rocprofv3 confirms `fmha_*`/`*ck_*` (engaged). Gate:
measured win over Triton on bf16 AND parity AND engaged.

## Alternatives / cross-links
./triton.md (editable) · ./tilelang.md · ./asm.md (peak) · ./aiter.md (ships CK) ·
../../attention_decode_paged/backends/ck.md · ../../mla_attention/backends/ck.md ·
../overview.md · CK language deep-dive: `languages/composable_kernel/fmha_template.md`.

## Sources
- CK-Tile FlashAttention-v2 walkthrough (~100 lines of CK-Tile = FA): https://rocm.blogs.amd.com/software-tools-optimization/ck-tile-flash/README.html
- FMHA example location + codegen `-d fwd/fwd_splitkv/batch_prefill`, `ENABLE_CK` gate: on-box `ROCm/aiter@a6bb499375849eec45d68c5ccaebc8865fd422c0:aiter/ops/mha.py` (lines 98, 390–394, 1062, 1984).
- CK default backend, head_dim ≤256 fwd+bwd, MI200x–MI355x+RDNA3/4 support: https://github.com/Dao-AILab/flash-attention
- AITER FA 1.2–4.4× TPS (vendor, MI300X/325X/355X, 2026-01-29): https://vllm.ai/blog/2026-02-27-rocm-attention-backend
