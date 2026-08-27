---
title: attention_prefill_fmha — tuning
kind: operator_overview
operator: attention_prefill_fmha
gens: [gfx90a, gfx942, gfx950, gfx1250]
dtypes: [bf16, fp16, fp8_e4m3_fnuz, fp8_e4m3]
regimes: [prefill]
updated: 2026-07-14
sources:
  - https://rocm.blogs.amd.com/software-tools-optimization/ck-tile-flash/README.html
  - https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/workload.html
  - https://rocm.blogs.amd.com/ecosystems-and-partners/rocm-tilelang-kernel/README.html
  - https://github.com/triton-lang/triton/blob/main/third_party/amd/backend/compiler.py
  - https://vllm.ai/blog/2026-02-27-rocm-attention-backend
---

# attention_prefill_fmha — tuning

How to make the prefill FlashAttention forward fast on CDNA3 (gfx942 / MI300X) and CDNA4 (gfx950 /
MI350X). The kernel is **two chained GEMMs around an online softmax** (`S=QKᵀ`, then `O=PV`), so the
tuning problem is different from a single dense GEMM: you co-schedule two matrix-core loops while
keeping the running max/sum in registers and never materializing the S=[sq,sk] score matrix.

## The decision first
The fastest *production* prefill FMHA on MI300X is the CK-Tile FMHA kernel (the backend behind
flash-attention ROCm and the default in vLLM/sglang via aiter). **Do not hand-tune attention before you
have confirmed which backend is live** — see [overview.md](overview.md) and the backend cards. The
levers below apply when you are authoring/tuning a Triton/TileLang/CK kernel, or pinning a config.

**Backend ranking (vLLM ROCm, Feb 2026):** ROCM_AITER_FA (recommended/auto) > ROCM_AITER_UNIFIED_ATTN
(~within 5%) > TRITON_ATTN > ROCM_ATTN. Concrete TPS uplift, ROCM_AITER_FA vs legacy ROCM_ATTN (64/128
req): **MI300X 3.82×/2.65×, MI325X 4.36×/3.12×, MI355X 3.61×/2.88×**; TPOT **2.8–4.6×** faster. So the
first "tune" is just picking ROCM_AITER_FA. **HipKittens** forward is the academic SOTA (beats AITER asm
1.0–2.1× in ~500 LoC) — see hipkittens.

## The shared lever set (every backend exposes some of these)
| lever | what it controls | CDNA3 starting point | source |
|---|---|---|---|
| `BLOCK_M` (Q rows/block, `kM0`) | occupancy vs Q-reuse — the #1 lever | 128 (64 for small heads / register pressure) | CK fmha_template; TileLang blog |
| `BLOCK_N` (KV tile, `kN0`) | inner-loop KV chunk; smaller → more masking skip on causal | 64 (TileLang FA optimal: `block_M=128, block_N=32`) | TileLang blog |
| head-dim tile `kK0/kK1` | Q·K and P·V contraction depth | 64/128; head_dim ≤256 supported | CK fmha_template |
| `num_warps` (wave64!) | threads = warps×64 | 4 (8 only if VGPR-light); TileLang FA uses 512 threads = 8 warps | triton_amd/knobs |
| `num_stages` | software-pipeline depth of the two GEMMs | **1** for fused FA (not 2-4) | triton_amd/knobs |
| `matrix_instr_nonkdim` (MFMA) | 16×16×16 vs 32×32×8 | **16** (lower power, higher achievable FLOPs) | triton_amd/knobs; asm_mfma/overview |
| `waves_per_eu` | force occupancy by trimming VGPRs | 1-3 (attention is register-heavy) | triton_amd/knobs |
| `schedule_hint` (Triton) | FA-aware instruction scheduling | `attention` / `memory-bound-attention` | triton_amd/knobs |
| causal/SWA/alibi mask trait | skip upper-triangle KV tiles | always specialize causal separately | CK fmha_template |
| `OPTIMIZE_EPILOGUE=1` | store O in MFMA layout (2-GEMM fusion) | ON | ROCm workload guide |

## CDNA3 vs CDNA4 (the budgets that change the tile)
- **LDS budget**: 64 KB/CU on gfx942, **160 KB/CU on gfx950**. FA stages Q (and sometimes K/V) tiles
  through LDS; on gfx942 a `BLOCK_M=128, head_dim=128` Q tile + K/V double-buffer is already tight, so
  large `BLOCK_N` or `num_stages>1` silently drops occupancy to 1 wg/CU. On gfx950 you have ~2.5× the
  LDS, so larger KV tiles / deeper prefetch are affordable.
- **Register split**: with one wave/SIMD the 512 VGPR split into ~256 VGPR + 256 AGPR (the MFMA
  accumulator lives in AGPR). The O accumulator + running stats are the dominant register consumers in
  FA — this is why `BLOCK_M` is the occupancy knob and why attention favors `waves_per_eu`≈1-3.
- **Async copy / direct-to-LDS**: `qr_ks_vs_async` (CK) and `knobs.amd.use_async_copy` (Triton) stream
  K/V global→LDS via `buffer_load` with no VGPR staging. gfx950-default, experimental on gfx942.
- **fp8 path**: fp8 QKV uses the fp8 WarpGemm + per-tile scale (16×16×32 / 32×32×16). FNUZ on CDNA3,
  OCP on CDNA4 — wrong dialect is off by exactly 2× (silent garbage). See [numerics.md](numerics.md).

## Grid sizing (fill 304/256 CUs)
Fold `batch × heads × ceil(sq/BLOCK_M)` into the grid; aim for ≥1024 workgroups so all 304 CUs (MI300X,
8 XCDs × 38) stay fed. For short sequences with few heads the grid starves — that is when split-Q /
flash-decoding style KV-splitting (the decode trick) starts to matter even in prefill.

## Head-dim specifics
- **64 / 128**: the common LLM head dims; 128 is the CK reference (`kK0=128`).
- **192 / 256**: CDNA3/CDNA4-relevant (e.g. some long-context and MLA-adjacent shapes). CK FMHA forward
  supports **head_dim ≤ 256**; above that you need the Triton FA backend (arbitrary head dim). Larger
  head dim → larger Q/O tiles → more LDS/VGPR pressure → drop `BLOCK_M` to keep occupancy.

## Autotune recipe (Triton/TileLang)
TileLang's FA example autotunes 108 candidates over `(block_M, block_N, num_stages, threads, num_split_q,
coalesced widths, GemmWarpPolicy)` in ~1 s and reports the winner (block_M=128, block_N=32, threads=512
on MI300X). Triton: search `BLOCK_M∈{64,128}`, `BLOCK_N∈{32,64,128}`, `num_warps∈{4,8}`,
`num_stages=1`, `waves_per_eu∈{1,2,3}`, `matrix_instr_nonkdim=16`, `schedule_hint=attention`; prune by
LDS ≤ 64 KB (gfx942) / 160 KB (gfx950). Bake the winner per shape (do not autotune on the hot path) —
tuned tables are ROCm/Triton-build-specific (sourcing rule #2).

## How to verify a tune helped
Isolated FMHA bench at your exact `(B,H,sq,sk,head_dim,causal,dtype)`, median of ≥3 warm reps, vs the
current backend; ISA check (`AMDGCN_ENABLE_DUMP=1`): want `global_load_dwordx4`, LDS `ds_read_b128`,
`v_mfma_*16x16`, no `scratch_` spills. Then e2e via `--attention-backend` swap + greedy temp=0 parity.

## Sources
- CK-Tile FMHA pipeline/knobs (qr_ks_vs(_async), kM0/kK0/kK1, head_dim≤256, fp8 WarpGemm): https://rocm.blogs.amd.com/software-tools-optimization/ck-tile-flash/README.html
- MI300X workload optimization (≥1024 grid, mfma_16x16, OPTIMIZE_EPILOGUE, 2-GEMM fusion): https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/workload.html
- TileLang FA autotune (108 configs, block_M=128/block_N=32/threads=512, MI300X): https://rocm.blogs.amd.com/ecosystems-and-partners/rocm-tilelang-kernel/README.html
- Triton AMD knobs (num_stages=1 for FA, schedule_hint, waves_per_eu, matrix_instr_nonkdim): https://github.com/triton-lang/triton/blob/main/third_party/amd/backend/compiler.py
- LDS 64 KB (CDNA3) / 160 KB (CDNA4), 256/512 VGPR split: see `hardware/cdna3_mi300/`, `hardware/cdna4_mi350/`, `languages/asm_mfma/overview.md`.
- Backend ranking + ROCM_AITER_FA vs ROCM_ATTN TPS (MI300X 3.82×/2.65×, MI325X 4.36×/3.12×, MI355X 3.61×/2.88×; TPOT 2.8–4.6×), HipKittens fwd academic SOTA (vendor, Feb 2026): https://vllm.ai/blog/2026-02-27-rocm-attention-backend ; HK arXiv 2511.08083.
