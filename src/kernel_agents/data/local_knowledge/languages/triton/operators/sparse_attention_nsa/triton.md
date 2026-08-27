---
title: sparse_attention_nsa on Triton — SOTA card
kind: sota_card
operator: sparse_attention_nsa
backend: triton
gens: [gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz]
regimes: [prefill, decode]
status: sota
updated: 2026-06-08
sources:
  - https://github.com/fla-org/native-sparse-attention
  - https://arxiv.org/html/2508.18224v1
  - https://fergusfinn.com/blog/deepseek-v4-flash-mi300x/
  - ROCm/aiter@a6bb499375849eec45d68c5ccaebc8865fd422c0
---

# sparse_attention_nsa × Triton

## TL;DR
Triton is the **SOTA and effectively only portable** NSA path on AMD. Two flavors: (a) the **research
reference** `fla-org/native-sparse-attention` (`parallel_nsa`, online top-k, fused selected+sliding) and
Flash Sparse Attention — CUDA-targeted but Triton runs on MI300X via triton_amd with the usual AMD
tuning; (b) the **production** DeepSeek sparse-MLA path shipped in aiter's Triton kernels
(`unified_attention_sparse_mla`, `pa_mqa_logits`, `fp8_mqa_logits`) and wired in sglang's `nsa_backend.py`
/ `nsa/nsa_indexer.py`. There is **no hand-tuned CK/asm NSA kernel** publicly available on AMD as of
2026-06; the production path is Triton + some HIP glue, with gfx942 falling back to generic Triton where
AITER's tuned path is missing or broken.

## SOTA implementation(s)
| impl | source | gens/dtypes | measured perf | when best |
|---|---|---|---|---|
| aiter sparse-MLA Triton | `ROCm/aiter@a6bb49937:aiter/ops/triton/attention/unified_attention_sparse_mla.py`, `pa_mqa_logits.py`, `fp8_mqa_logits.py` | gfx942/950; bf16/fp16/fp8 | DeepSeek-V4 MI300X: sparse MLA among most-expensive layers; bookkeeping fix +8.6% (2485→2699 tok/s/GPU) | DeepSeek-V3.2/V4 sparse MLA serving |
| sglang NSA backend | `sgl-project/sglang:python/sglang/srt/layers/attention/nsa_backend.py`, `nsa/nsa_indexer.py` | gfx942/950 | DeepSeek sparse attn; ROCm coverage maturing | sglang serving (`--attention-backend nsa`/`dsa`) |
| FLA `parallel_nsa` (reference) | `fla-org/native-sparse-attention` | CUDA-written; runs on MI300X via Triton ROCm | research-grade; not AMD-tuned | training/research, custom NSA |
| Flash Sparse Attention | `arXiv 2508.18224` | CUDA-targeted; portable | alternative selected-branch kernel design | when reference selected-branch stalls |

## Config space / knobs
- `block_size=64` (selected/compressed), `window_size`, `top_k` — match the model config.
- AMD Triton: `BLOCK_N == block_size`, `matrix_instr_nonkdim=16`, `num_warps=4`, `num_stages=1`,
  `waves_per_eu=2–3`, `schedule_hint=attention`, `knobs.amd.use_buffer_ops=ON` (indexed gather).
- fp8 indexer: **fnuz** on gfx942. See [tuning.md](tuning.md).

## Numerics / parity
Discrete top-k selection (selection-overlap gate), fp8 indexer (fnuz accuracy gate), three-branch fp32
gated sum. See [numerics.md](numerics.md).

## Integration (rebind seam)
- sglang: `--attention-backend nsa` (or `dsa`/`dsv4`); `SGLANG_OPT_USE_AITER_INDEXER` toggles the aiter
  indexer. `nsa_backend.py` import block is the "which kernel ran" surface.
- The `@triton.jit` selected/indexer kernels are clean Python rebind seams — overlay a tuned config and
  verify via `TRITON_PRINT_AUTOTUNING=1`.

## Pitfalls & anti-patterns
- **gfx942 fallback cliff:** AITER prefill/sparse MQA-logits broken on CDNA3 → dispatch refused → generic
  Triton (several × slower). Paged MQA logits / sparse MLA prefill+decode were *missing entirely* on
  gfx942 → ROCm helper → Triton.
- **HIP-graph capture** of ragged index tensors → corruption; use static capture-safe metadata.
- Reference FLA/FSA kernels carry CUDA assumptions (`device='cuda'`, NVIDIA tile defaults) — re-tune for
  CDNA3 (wave64, 64 KB LDS, mfma_16x16).
- Selection done densely (materializing the score matrix) defeats the point — use online top-k.

## How to verify
Prefill time scales with `top_k·block_size`, not `seq`; selection-overlap vs reference indexer; greedy
temp=0 + gsm8k-style eval (sparse paths have shown eval regressions). `AITER_LOG_MORE=1` to confirm the
tuned vs fallback path.

## Alternatives / cross-links
[overview.md](overview.md) · hip · tilelang · languages: triton_amd ·
core: mla_attention · sliding_window_attention · backend: aiter · sglang_kernels.

## Sources
- aiter sparse-MLA / mqa-logits kernels: `ROCm/aiter@a6bb49937:aiter/ops/triton/attention/` (on-box).
- sglang NSA backend + indexer: https://github.com/sgl-project/sglang (`layers/attention/nsa_backend.py`, `nsa/nsa_indexer.py`).
- FLA NSA reference: https://github.com/fla-org/native-sparse-attention
- FSA: https://arxiv.org/html/2508.18224v1
- DeepSeek-V4 MI300X bring-up (+8.6%, gfx942 fallback): https://fergusfinn.com/blog/deepseek-v4-flash-mi300x/
