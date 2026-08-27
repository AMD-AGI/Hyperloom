---
title: mla_attention — tuning
kind: operator_overview
operator: mla_attention
gens: [gfx942, gfx950, gfx1250]
dtypes: [bf16, fp16, fp8_e4m3_fnuz]
regimes: [prefill, decode]
updated: 2026-07-14
sources:
  - ROCm/aiter@b467ce342:aiter/mla.py
  - https://rocm.blogs.amd.com/software-tools-optimization/aiter-mla/README.html
  - https://vllm.ai/blog/2026-02-27-rocm-attention-backend
---

# mla_attention — tuning

MLA tuning is two distinct problems: **absorbed decode** (bandwidth-bound MQA over the latent — the
17× kernel) and **prefill** (GEMM-bound, often unabsorbed). Decide the form first, then tune.

## Decision: absorbed vs unabsorbed
- **Decode → absorbed (weight-absorbed).** Fold `Wuk` into `q_nope` and `Wuv` into the output so the
  layer is MQA on the 512-wide latent + 64-wide RoPE. Minimal KV bandwidth. This is `mla_decode_fwd`.
- **Prefill → often unabsorbed.** Materialize K/V from the latent then run MHA; long sq amortizes the
  up-projection. `mla_prefill_fwd` / `mla_prefill_ps_fwd` (persistent). Tune prefill independently of
  decode — the tile shapes don't carry over.

## The latent + RoPE split (the structural lever)
The score is `q_nope·c_KVᵀ` over `kv_lora_rank` **plus** `q_rope·k_ropeᵀ` over `qk_rope_head_dim`. The
Triton kernel splits `BLOCK_DMODEL=512` (latent) + `BLOCK_DPE` (rope dim, 0 if no decoupled RoPE). The
512-wide latent contraction is the dominant cost; on the asm path it maps onto the MFMA pipe as one big
MQA reduction. **head_dim here is effectively 512+64 = 576** — far above the 256 MHA cap — which is *why*
MLA needs a dedicated kernel, not the generic FMHA path.

## Decode levers (`mla_decode_fwd`)
- **`num_kv_splits` (split-KV)** — auto-picked by `get_meta_param` from `(batch, total_kv, nhead,
  max_seqlen_q, dtype)`. **Leave `None` (auto)** — the comment in source literally says "for experts
  only!!!". A Triton stage-2 (`_fwd_kernel_stage2_asm`) combines partials. More splits when batch is
  small (fill CUs). The CU-occupancy score scales occupancy by `max(tg_factor, wg_per_split)` — on
  **gfx1250** `wg_per_split=2` for `nhead==128` (two head-group WGs per split). For fp8 the split count is
  clamped to a per-head `min_block_n` table keyed on `nhead*max_seqlen_q`:
  `{8:64, 16:128, 24:128, 32:128, 48:64, 64:64, 128:32, 256:32, 384:32, 512:32}`.
- **`mgc` (the kernel's internal grid/merge constant)** is auto-set by `(nhead, dtype, max_seqlen_q)`:
  64 for `nhead==16, sq==1`; 32 for `nhead==128 fp8/fp8` or `nhead==64 bf16/bf16 sq==1`; else 16. You
  don't set this — it documents that the kernel specializes per `(nhead, dtype)`.
- **`page_size` / `nhead_kv`** — decode contract is `nhead_kv==1` (MQA); `page_size==1` uses the unpaged
  latent representation (fastest).
- **persistent mode** (`work_meta_data`/`work_indptr` set, or `SGLANG_AITER_MLA_PERSIST=1`) keeps the
  kernel resident across the decode loop — amortizes launch at small batch.
- **fp8 latent / fp8 KV** (`q_scale`, `kv_scale`) — halves bandwidth; accuracy-gate. Note aiter MLA
  decode does **not** support fp8 KV-cache in vLLM upstream (per vLLM ROCm blog).
- **context-parallel (round-robin)** — pass the GLOBAL `g_kv_indptr` + `cp_world_size` / `cp_rank` to
  shard a sequence round-robin across ranks (local kv idx `j` → global pos `j*W+r` for the causal mask);
  the C++ selects the dedicated `cprr` asm kernel (**gfx950** only). `cp_world_size==1` disables it.
- **DeepSeek-V4 sparse decode** (`mla_decode_fwd_v4_nm`, **gfx950 + gfx1250**) — fp8 Q/KV + bf16 RoPE,
  **required** `sink`, gqa≤128, `sm_scale` hardcoded `1/sqrt(512)`. `num_kv_splits=None` reuses V3's
  `get_meta_param` heuristic (with `tg_factor=ceil(num_heads/64)`); multi-split merges via the same Triton
  stage-2. gfx942 has expanded fp8 asm only behind `AITER_ENABLE_EXPERIMENTAL`.

## Prefill levers (`mla_prefill_fwd` / `_ps_fwd`)
GEMM-bound: tile the up-projection + the two attention GEMMs; persistent variant amortizes launch for
many short prefills. `mla_prefill_reduce` combines partials. Tune like a 2-GEMM FMHA (see
../attention_prefill_fmha/tuning.md) but with head_dim 576.

## CDNA3 vs CDNA4
- AITER MLA vs Triton-MLA **TPS: MI300X 1.33×, MI325X 1.41×, MI355X 1.52×**; TPOT **1.2–1.6×** (vLLM
  ROCm, Feb 2026). Backend identity: `ROCM_AITER_MLA ≈ ROCM_AITER_TRITON_MLA` (share the asm decode
  kernel) > `TRITON_MLA`.
- gfx950 uses the **AITER assembly MHA prefill** for MLA, so `ROCM_AITER_MLA` matches/beats
  `ROCM_AITER_TRITON_MLA` and gets the **best TTFT** on MI355X. On **gfx942**, the `ROCM_AITER_TRITON_MLA`
  variant shows **+2–3% higher TPS** (vendor). Bake off both per gen.
- LDS 64 KB (gfx942) / 160 KB (gfx950): the 512-wide latent tile is LDS-heavy; gfx950's larger LDS helps.
- fp8: FNUZ on gfx942, OCP on gfx950 (wrong dialect off by 2×).
- **gfx1250 (MI400)**: ships the mla asm + V4 sparse-decode HSACO (`hsa/gfx1250/mla`, `mla_v4`); the
  auto-split rule launches 2 head-group WGs per (batch, split) at `nhead==128`. Bake off per gen.

## Framework knobs (serving)
sglang: `--attention-backend aiter`, `SGLANG_ROCM_FUSED_DECODE_MLA=1` (fused MLA decode + RoPE),
`SGLANG_AITER_MLA_PERSIST=1`. vLLM: `VLLM_ROCM_USE_AITER=1` + `VLLM_ROCM_USE_AITER_MLA=1`,
`--attention-backend ROCM_AITER_MLA` (auto-selected, recommended for all workloads).

## How to verify a tune helped
`AITER_LOG_MORE=1` to confirm the asm MLA kernel (not Triton) fires; isolated `mla_decode_fwd` bench at
`num_heads∈{16,64,128}`, `kv_lora_rank=512`, `qk_rope_head_dim=64`, context/batch sweep, median ≥3 reps;
e2e TPOT + accuracy gate (MLA has shown eval regressions).

## Sources
- `mla_decode_fwd` / `mla_prefill_fwd` knobs (num_kv_splits auto "experts only", mgc per (nhead,dtype), persistent mode, q_scale/kv_scale, page_size/nhead_kv): on-box `ROCm/aiter@b467ce342:aiter/mla.py`.
- absorbed decode = MQA, 17×: https://rocm.blogs.amd.com/software-tools-optimization/aiter-mla/README.html
- AITER MLA vs Triton-MLA TPS (MI300X 1.33× / MI325X 1.41× / MI355X 1.52×, TPOT 1.2–1.6×), gfx942 Triton-MLA +2–3% TPS / gfx950 ROCM_AITER_MLA best TTFT, fp8 KV unsupported in vLLM upstream (vendor, Feb 2026): https://vllm.ai/blog/2026-02-27-rocm-attention-backend
