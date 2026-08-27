---
title: mla_attention — overview
kind: operator_overview
operator: mla_attention
gens: [gfx942, gfx950, gfx1250]
dtypes: [bf16, fp16, fp8_e4m3_fnuz, fp8_e4m3]
regimes: [prefill, decode]
updated: 2026-07-14
sources:
  - ROCm/aiter@b467ce342:aiter/mla.py
  - https://rocm.blogs.amd.com/software-tools-optimization/aiter-mla/README.html
  - https://vllm.ai/blog/2026-02-27-rocm-attention-backend
  - https://arxiv.org/abs/2405.04434
---

# mla_attention  (DeepSeek Multi-head Latent Attention)

## TL;DR
MLA (DeepSeek-V2/V3) compresses the KV-cache into a **low-rank latent** (`kv_lora_rank`, e.g. 512) plus a
small **decoupled RoPE** part (`qk_rope_head_dim`, e.g. 64), so the cache stores ~1/10 the bytes of MHA.
The decode-phase trick that makes it fast on AMD is **matrix absorption (weight-absorbed form)**: fold
the KV up-projection weights into Q and into the output, so the layer runs as **MQA** (one shared KV
head) directly on the latent — collapsing bandwidth and letting a hand-tuned asm kernel saturate the MFMA
pipe. On MI300X aiter's `mla_decode_fwd` reports **up to 17×** vs naive decode; the AITER MLA serving
backends give **1.2–1.6× faster TPOT** vs Triton MLA.

## Math contract
- **Cache**: latent `c_KV ∈ [tokens, kv_lora_rank]` (e.g. 512) + RoPE key `k_rope ∈ [tokens,
  qk_rope_head_dim]` (e.g. 64). Total cached width per token = `kv_lora_rank + qk_rope_head_dim` (576).
- **Absorbed decode**: `q` is `[B·q_seqlen, num_heads, kv_lora_rank + qk_rope_head_dim]` (the nope part is
  pre-multiplied by the absorbed `Wuk`); attention runs MQA over the latent; output `o` is
  `[B·q_seqlen, num_heads, kv_lora_rank]`, then the absorbed `Wuv` maps it to `v_head_dim`.
- **Two GEMMs split by RoPE**: the score is `q_nope·c_KVᵀ` (over `kv_lora_rank`) **plus**
  `q_rope·k_ropeᵀ` (over `qk_rope_head_dim`) — the Triton kernel splits `BLOCK_DMODEL=512` (latent) +
  `BLOCK_DPE` (rope). fp32 online-softmax accumulate.

## Absorbed vs unabsorbed (the key decision)
- **Absorbed (weight-absorbed) decode** = MQA on the latent: minimal KV bandwidth, the fast path. This is
  what `mla_decode_fwd` does.
- **Unabsorbed** = materialize full K/V from the latent then run MHA: more FLOPs/bandwidth, used in some
  **prefill** paths where the absorbed form is less advantageous (long sq makes the up-projection
  amortize). Prefill (`mla_prefill_fwd`) and decode are tuned separately.

## Shape regimes
- **Decode** (`sq=1`): the headline — bandwidth-bound MQA over the latent; `mla_decode_fwd` + splitKV.
- **Prefill** (long sq): `mla_prefill_fwd` (and a persistent variant `mla_prefill_ps_fwd`); GEMM-bound.
- DeepSeek shapes: `num_heads` 16/64/128, `kv_lora_rank=512`, `qk_rope_head_dim=64`, `v_head_dim=128`.
- **DeepSeek-V4 sparse decode** (`mla_decode_fwd_v4_nm`): fp8 Q/KV + bf16 RoPE with a **required**
  attention `sink`, gqa≤128; gen-specific asm on **gfx950 + gfx1250** (MI350/MI400), not gfx942. See
  [tuning.md](tuning.md).

## Where it matters (Amdahl)
For DeepSeek-class models MLA is the attention hot path; decode MLA is a top-1 decode kernel. The 17×
figure is the isolated decode-kernel speedup; e2e the AITER MLA backends give 1.2–1.6× TPOT and up to
1.5× TPS vs Triton MLA. `mla_attention` is distinct from ../gqa_mqa_attention/overview.md (which
broadcasts a real KV head) — MLA attends a *compressed latent*.

## Backend landscape (→ SOTA cards)
| backend | status | card |
|---|---|---|
| aiter | 🟢 sota (asm `mla_decode_fwd`, 17×) | [aiter.md](aiter.md) |
| triton | 🟡 (reference + fallback; `mla_decode.py`) | triton |
| ck | 🟡 (CK-Tile MLA; from-source) | ck |
| hip | 🟡 (vLLM custom; mostly routes to AITER MLA) | hip |
| fa_rocm | ⚪/🟡 (no dedicated MLA; use AITER MLA) | fa_rocm |

## Fusion neighbors
Fused KV-down-projection + RoPE + latent-cache write + quant (pre); fused decode + RoPE
(`SGLANG_ROCM_FUSED_DECODE_MLA`); persistent decode (`SGLANG_AITER_MLA_PERSIST`). See
[fusion.md](fusion.md).

## Numerics
Matrix absorption is **algebraically equivalent** to standard MLA (parity-safe in bf16); fp8 latent /
fp8 KV introduces quant error — accuracy-gate (AITER MLA has shown eval regressions). See
[numerics.md](numerics.md).

## How to bench
Isolated `mla_decode_fwd` (AI-Developer-Hub notebook) at `num_heads∈{16,64,128}`, `kv_lora_rank=512`,
`qk_rope_head_dim=64`, context 1k–8k, batch 1..512; e2e via `--attention-backend` MLA variant + TPOT.

## Sources
- aiter MLA decode/prefill signatures, absorption, splitKV, fp8 (`q_scale`/`kv_scale`), persistent mode: on-box `ROCm/aiter@b467ce342:aiter/mla.py`.
- 17× decode + matrix absorption (vendor, MI300X, 2025-03): https://rocm.blogs.amd.com/software-tools-optimization/aiter-mla/README.html
- AITER MLA 1.2–1.6× TPOT / 1.5× TPS vs Triton MLA (vendor, 2026-01-29): https://vllm.ai/blog/2026-02-27-rocm-attention-backend
- MLA definition (DeepSeek-V2): https://arxiv.org/abs/2405.04434
