---
title: sliding_window_attention on CK-Tile — SOTA card
kind: sota_card
operator: sliding_window_attention
backend: ck
gens: [gfx90a, gfx942, gfx950]
dtypes: [bf16, fp16]
regimes: [prefill, decode]
status: sota
updated: 2026-06-08
sources:
  - https://github.com/Dao-AILab/flash-attention
  - https://blog.vllm.ai/2026/02/27/rocm-attention-backend.html
  - ROCm/composable_kernel:example/ck_tile/01_fmha/mask.hpp
---

# sliding_window_attention × CK-Tile

## TL;DR
CK-Tile FMHA is the **reliable, default SWA path on ROCm**. Its mask layer natively supports FA-style
sliding windows (`window_generic` with `left`/`right`) plus attention sinks, and the mask encodes the
KV-block early-out so block-skipping is automatic. On vLLM/sglang the practical recipe for SWA models is
`VLLM_USE_TRITON_FLASH_ATTN=0` (select CK) — historically chosen precisely because the Triton FA backend
lacked SWA. Use CK for SWA unless you need a feature only the Triton backend has (fp8 / arbitrary head
dim / ALiBi), or you're rewriting the kernel.

## SOTA implementation(s)
| impl | source | gens/dtypes | measured perf | when best |
|---|---|---|---|---|
| CK-Tile FMHA + `window_generic` mask | `ROCm/composable_kernel:example/ck_tile/01_fmha` (mask.hpp, generate.py) | gfx90a/942/950; bf16/fp16; head_dim ≤256 | default FA backend; no public per-shape SWA TFLOPS quoted — measure on-box | general SWA prefill/decode, causal/symmetric window, sinks |
| flash-attention ROCm CK backend | `Dao-AILab/flash-attention` (ROCm), `window_size=(left,right)` | as above | the path vLLM uses with `VLLM_USE_TRITON_FLASH_ATTN=0` | SWA models (Mistral/Gemma/Qwen2/3) at half precision |

No hand-tuned asm SWA kernel is publicly documented as of 2026-06; CK-Tile is the SOTA SWA path.

## Config space / knobs
- Select the mask: `window_generic` (FA `left/right`) or top-left/bottom-right + `sink_size`; codegen via
  the FMHA `generate.py`. The mask builder `make_generic_attention_mask_coordinates_from_lr_window`
  already gives the per-tile loop range (block-skip).
- Tiling as full FMHA: `M/N PerBlock`, `M/N PerXDL` (mfma_16x16), AK1/BK1 ≥128-bit loads, ≥1024 wgs.
- See composable_kernel fmha_template.md / knobs.md.

## Numerics / parity
fp32 online-softmax; window edge inclusive of `left` past tokens; CK splits an even window symmetrically
(`left=W/2`) — match the HF `sliding_window` definition. Soft-cap before mask. See [numerics.md](numerics.md).

## Integration (rebind seam)
- vLLM: `VLLM_USE_TRITON_FLASH_ATTN=0` selects CK (the documented SWA fix). On V1 confirm the banner —
  the legacy flag can be ignored; select the backend explicitly.
- flash-attention ROCm: CK is the default build; pass `window_size=(left,right)` to `flash_attn_func`.
- The CK FMHA forward callable is the capture/rebind seam for an authored SWA replacement.

## Pitfalls & anti-patterns
- composable_kernel moved into `ROCm/rocm-libraries` (standalone repo deprecated) — pin accordingly.
- CK FA is fp16/bf16 only (no fp32) and head_dim ≤256; above that you need the Triton backend.
- vLLM V1 may report "Triton Attention backend" even with the legacy flag set — verify the active path.
- Even window ⇒ symmetric split; if the model wants causal-only SWA pass `left=W-1, right=0` explicitly.

## How to verify
Greedy temp=0 parity vs dense band-mask reference (≥10 prompts, some longer than `W`); isolated FMHA
bench vs the Triton backend at the same `(b,h,seq,d,window)`; check wall-clock scales with `window`, not
`seq`; confirm the CK backend banner in the server log.

## Alternatives / cross-links
[overview.md](overview.md) · triton · [flash_attention_rocm.md](fa_rocm.md) ·
aiter · core: attention_prefill_fmha · language: composable_kernel.

## Sources
- CK window/sink mask: `ROCm/composable_kernel:example/ck_tile/01_fmha/mask.hpp` (on-box via aiter 3rdparty).
- FA window_size API + CK default on ROCm: https://github.com/Dao-AILab/flash-attention
- CK for SWA (`VLLM_USE_TRITON_FLASH_ATTN=0`), V1 banner caveat: https://blog.vllm.ai/2026/02/27/rocm-attention-backend.html
