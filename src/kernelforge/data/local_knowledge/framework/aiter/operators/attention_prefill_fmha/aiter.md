---
title: attention_prefill_fmha on aiter — SOTA card
kind: sota_card
operator: attention_prefill_fmha
backend: aiter
gens: [gfx942, gfx950, gfx1250]
dtypes: [bf16, fp16, fp8_e4m3_fnuz]
regimes: [prefill]
status: sota
updated: 2026-07-14
sources:
  - ROCm/aiter@b467ce342:aiter/ops/mha.py
  - https://github.com/ROCm/aiter
  - https://vllm.ai/blog/2026-02-27-rocm-attention-backend
---

# attention_prefill_fmha × aiter

## TL;DR (one-line decision)
> aiter is the **default, production prefill attention** on AMD serving: `aiter.flash_attn_func` /
> `mha_batch_prefill` dispatches to a hand-tuned **asm/CK** MHA prefill kernel (with a Triton fallback
> when `ENABLE_CK` is unset or no CK instance exists for the shape). It is what sglang
> `--attention-backend aiter` and vLLM `ROCM_AITER_FA` actually call. Backend ranking (vLLM ROCm,
> Feb 2026): **ROCM_AITER_FA > ROCM_AITER_UNIFIED_ATTN (~within 5%) > TRITON_ATTN > ROCM_ATTN**.
> Concrete TPS, ROCM_AITER_FA vs ROCM_ATTN (64/128 req): **MI300X 3.82×/2.65×, MI325X 4.36×/3.12×,
> MI355X 3.61×/2.88×**; TPOT **2.8–4.6×** faster. Use it as the default; drop to CK-Tile directly only
> for a specific feature gap, or to TileLang/Triton for editability. **HipKittens** forward is the
> *academic* SOTA (beats AITER asm 1.0–2.1× in ~500 LoC) — cross-link ./hipkittens.md.

## SOTA implementation(s)
| impl | source (`repo@commit:path`) | gens / dtypes / shapes | measured perf (`value @ hw, ROCm/lib, date`) | when it's best |
|---|---|---|---|---|
| `aiter.flash_attn_func` (asm/CK FA-2) | `ROCm/aiter@b467ce342:aiter/ops/mha.py:2533` | gfx942/950; bf16/fp16/fp8 FNUZ; head_dim ≤256 (CK) / arbitrary (Triton); MQA/GQA | **ROCM_AITER_FA vs ROCM_ATTN TPS (64/128 req): MI300X 3.82×/2.65×, MI325X 4.36×/3.12×, MI355X 3.61×/2.88×; TPOT 2.8–4.6×** @ vLLM ROCm blog, Feb 2026 (vendor) | default MHA prefill, dense q/kv |
| `aiter.mha_batch_prefill` (paged batch prefill) | `aiter/ops/mha.py:3600` | gfx942/950; bf16/fp16/fp8; paged-KV via `block_table`/`kv_page_indices` | chunked/paged prefill path in serving (vendor envelope above) | mixed-length batched prefill, paged-KV |
| aiter Triton FA (`ops/triton/attention/mha.py`) | same repo | fp8 / arbitrary head dim / ALiBi | feature-rich fallback (`if not ENABLE_CK`) | when the asm/CK path is missing for the shape |

**Real signature** — `flash_attn_func` is GQA/MQA-native and supports SWA + sinks (`aiter/ops/mha.py`):
```python
def flash_attn_func(q, k, v, dropout_p=0.0, softmax_scale=None, causal=False,
    window_size=(-1, -1, 0),   # (left, right, sink); -1 = infinite ctx, 0-slot = no sink
    bias=None, alibi_slopes=None, deterministic=True, return_lse=False,
    return_attn_probs=False, how_v3_bf16_cvt=0, num_splits=0,   # num_splits = split-K knob (0 = auto)
    cu_seqlens_q=None, cu_seqlens_kv=None, sink_ptr=None):
    ...
    if not ENABLE_CK:          # ~line 2607 — Triton fallback when CK disabled/missing
        from .triton.attention.mha import flash_attn_func as flash_attn_func_triton
```
`mha_batch_prefill` exposes the full feature set: `logits_soft_cap`, `window_size_left/right`,
`sink_size`, `q_descale/k_descale/v_descale` (PERTENSOR fp8) **or** `kv_block_descale`
(`[num_block, num_kv_head, 2]` KV_BLOCKSCALE mode — mutually exclusive with per-tensor), `block_table`.

## Config space / knobs
aiter selects the impl from its catalog; you do **not** hand-tune tiles at the call site.

| param | range / values | effect | default |
|---|---|---|---|
| `causal` | bool | bottom-right-aligned causal mask | False |
| `softmax_scale` | float | QK scale | `1/sqrt(d)` |
| `window_size` | `(left,right,sink)` | sliding-window + attention sink | `(-1,-1,0)` |
| `logits_soft_cap` (batch_prefill) | float ≥0 | tanh logit cap (Gemma-2) | 0 (off) |
| KV dtype | bf16 / fp8_e4m3 **FNUZ** | memory + bandwidth | bf16 |
| fp8 descale mode | PERTENSOR (`*_descale`) vs KV_BLOCKSCALE (`kv_block_descale`) | fp8 accuracy granularity | per-tensor |
| `ENABLE_CK` (env via jit/core) | 0/1 | CK kernel vs Triton fallback | image-dependent |

Framework gates: `VLLM_ROCM_USE_AITER=1` (**master**, required even when forcing a backend) +
`VLLM_ROCM_USE_AITER_MHA=1`; sglang `SGLANG_USE_AITER=1`; `SGLANG_AITER_FP8_PREFILL_ATTN` for fp8 prefill.

## Numerics / parity
fp32 online-softmax accumulate; fp8 prefill scales inputs. **fp8 dialect trap**: FNUZ on gfx942, OCP
e4m3 on gfx950 — wrong dialect off by ~2×. asm vs CK vs Triton reduction order differs → re-check
greedy/temp=0 parity after a backend swap. fp8 prefill is a task-accuracy gate (gsm8k/mmlu, not just
MSE). Head_dim ≤256 on the CK path; Triton handles arbitrary d. See [numerics.md](numerics.md).

## Integration (rebind seam)
- **sglang:** `--attention-backend aiter`; the dispatch surface is `aiter_backend.py`'s
  `from aiter import ...` import block (the literal "which AITER kernel ran").
- **vLLM:** `--attention-backend ROCM_AITER_FA` (+ master `VLLM_ROCM_USE_AITER=1`).
- Attention has a clean Python forward seam, so an authored replacement can be wired without editing
  site-packages (monkeypatch `aiter.flash_attn_func` / overlay import).
- **Verify it engaged:** `AITER_LOG_MORE=1` prints asm/CK vs Triton; rocprofv3 Top-N `fmha_*`/`*ck_*`.

## Pitfalls & anti-patterns
- `VLLM_ROCM_USE_AITER=1` is the **master** switch — forcing `ROCM_AITER_FA` without it is inert.
- gfx942 may fall back to Triton for newer variants → several × slower; confirm an asm/CK path exists for
  your shape (`AITER_LOG_MORE=1`).
- AITER CK can crash under HIP-graph capture for novel shapes (`device_gemm does not support this GEMM
  problem`, sglang #16025) → force Triton for that model.
- Per-tensor descale and `kv_block_descale` are **mutually exclusive** — passing both is a contract error.
- Don't `repeat_kv` before calling — aiter is GQA-native; replication wastes HBM (see gqa card).

## Worked example
Qwen2-72B prefill, GQA (64 q / 8 kv heads), d=128, bf16, causal, ctx=8192:
1. `VLLM_ROCM_USE_AITER=1 VLLM_ROCM_USE_AITER_MHA=1 --attention-backend ROCM_AITER_FA`.
2. `AITER_LOG_MORE=1` → confirm asm/CK FMHA (not Triton) for head_dim=128.
3. Bench prefill TTFT vs `--attention-backend ck` and `triton`; ROCM_AITER_FA should land ~2.6–4.4× TPS
   over ROCM_ATTN (per-GPU figures above) on GQA models.
4. If switching KV to fp8: set descale, confirm FNUZ on gfx942, then gsm8k accuracy gate.

## How to verify (bench + oracle)
`AITER_LOG_MORE=1` confirms asm/CK (not Triton) fired; isolated prefill bench vs CK-Tile/Triton at the
served shape; rocprofv3 Top-N → `fmha_*`/`*ck_*` names. Greedy temp=0 parity ≥10 prompts. Gate: win over
the alternative backend AND parity AND engaged.

## Alternatives / cross-links
./ck.md (the kernel under aiter's CK path) · ./triton.md · ./asm.md (peak) ·
../../attention_decode_paged/backends/aiter.md · ../../mla_attention/backends/aiter.md ·
../../gqa_mqa_attention/backends/aiter.md · `backends/sglang_kernels/attention_backends.md` ·
../overview.md.

## Sources
- On-box `ROCm/aiter@b467ce342:aiter/ops/mha.py` (`flash_attn_func`:2533, `mha_batch_prefill`:3600, `ENABLE_CK` fallback ~:2607; SWA/sink/softcap/fp8 descale modes, `num_splits`/`how_v3_bf16_cvt`/`return_attn_probs`).
- aiter as default kernel backend: https://github.com/ROCm/aiter
- ROCM_AITER_FA vs ROCM_ATTN TPS (MI300X 3.82×/2.65×, MI325X 4.36×/3.12×, MI355X 3.61×/2.88× @ 64/128 req; TPOT 2.8–4.6×), backend ranking ROCM_AITER_FA > ROCM_AITER_UNIFIED_ATTN > TRITON_ATTN > ROCM_ATTN (vendor, Feb 2026): https://vllm.ai/blog/2026-02-27-rocm-attention-backend
- HipKittens forward academic SOTA (beats AITER asm 1.0–2.1×): arXiv 2511.08083 — see ./hipkittens.md.
