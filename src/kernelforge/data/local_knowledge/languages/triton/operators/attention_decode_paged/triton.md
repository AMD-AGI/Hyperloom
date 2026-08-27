---
title: attention_decode_paged on Triton — SOTA card
kind: sota_card
operator: attention_decode_paged
backend: triton
gens: [gfx90a, gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz]
regimes: [decode]
status: competitive
updated: 2026-06-08
sources:
  - ROCm/aiter@a6bb499375849eec45d68c5ccaebc8865fd422c0:aiter/ops/triton/attention/pa_decode.py
  - ROCm/aiter@a6bb499375849eec45d68c5ccaebc8865fd422c0:aiter/ops/triton/attention/unified_attention.py
  - https://github.com/triton-lang/triton/blob/main/third_party/amd/backend/compiler.py
  - https://vllm.ai/blog/2026-02-27-rocm-attention-backend
---

# attention_decode_paged × Triton

## TL;DR (one-line decision)
> Triton paged-decode (`TRITON_ATTN`, aiter `pa_decode.py`) is the **universal fallback** and the
> **editable** decode kernel — it runs everywhere, any shape, with automatic **V1 (seqlen ≤8192) / V2
> (sequence-partitioned)** dispatch and flash-decoding. It is usually **slower than aiter asm / vLLM-HIP**
> decode, and the `ROCM_ATTN` fallback cliff (2.7–4.4× slower on an unsupported head size) is exactly this
> Triton kernel firing. Use it for portability, a custom decode variant, or as the correctness reference.

## SOTA implementation(s)
| impl | source (`repo@commit:path`) | gens / dtypes | measured perf | when it's best |
|---|---|---|---|---|
| aiter Triton paged-decode | `ROCm/aiter@a6bb49937:aiter/ops/triton/attention/pa_decode.py:34` | gfx90a/942/950; bf16/fp16/fp8 KV; any head dim | universal fallback; slower than asm/HIP decode (the `ROCM_ATTN` cliff is this kernel) | portability / editable / reference |
| Triton unified attention (decode side) | `aiter/ops/triton/attention/unified_attention.py` | as above | one kernel for chunked-prefill + decode | mixed batches (see chunked_prefill) |

**Real V1/V2 dispatch + fp8 scale granularity** (`pa_decode.py`):
```python
_SEQ_PARTITION_SIZE = 1024  # HIP
max_num_partitions = (max_seq_len + _SEQ_PARTITION_SIZE - 1) // _SEQ_PARTITION_SIZE
use_v1 = max_seq_len <= 8192 and (max_num_partitions == 1 or num_seqs * num_q_heads > 512)
if k_scale.numel() > 1:            # per-token fp8 KV (shape [blocks, kv_heads, block_size])
    ... paged_attn_decode_v1_per_token_quant(...)
else:                              # per-tensor fp8 KV (scalar scale)
    ...
```
Note this is `_SEQ_PARTITION_SIZE = 1024` for the **Triton** path (vs the asm `paged_attention_rocm`
partition of **256** — see aiter). fp8 supports **per-token** (scale shape
`[num_blocks, num_kv_heads, block_size]`) and **per-tensor** (scalar) KV quant.

## Config space / knobs
| param | range / values | effect | default |
|---|---|---|---|
| `_SEQ_PARTITION_SIZE` | 1024 (Triton) | seq partition for V2 flash-decoding | 1024 |
| V1/V2 select | `seqlen≤8192 & (1 partition or seqs·heads>512)` | launch count vs reduce overhead | auto |
| `BLOCK_N` (KV chunk) | 16 / 32 / 64 | KV streaming tile | 32 |
| fp8 scale mode | per-token vs per-tensor (`k_scale.numel()`) | fp8 accuracy granularity | per-tensor |
| `num_warps` | 4 (wave64) | parallelism | 4 |
| `num_stages` | **1** | fused decode pipeline | 1 |
| `matrix_instr_nonkdim` | 16 | MFMA 16×16 | 16 |
| `waves_per_eu` | 2–4 (decode is memory-bound → favor occupancy) | occupancy | 2 |
| `kpack` | 2 (gfx942) | LDS pack | 2 |

See [tuning.md](tuning.md) and `triton_levers/knobs.md`.

## Numerics / parity
fp32 online-softmax accumulate; V2 split reduce with per-split `m_i`. fp8 KV uses `q/k_descale` (per-token
or per-tensor) — **FNUZ on gfx942, OCP on gfx950**; wrong dialect off ~2×. Cross-backend bf16 tie-flips
benign. This kernel is a useful **decode oracle** for the asm path. See [numerics.md](numerics.md).

## Integration (rebind seam)
- **sglang:** `--attention-backend triton` / `--decode-attention-backend triton`.
- **vLLM:** `--attention-backend TRITON_ATTN`.
- The `@triton.jit` decode kernel is the Tier-C edit seam.
- **Verify it engaged:** rocprofv3 → `_paged_attn_decode_*` / `kernel_unified_attention_*` (Triton names),
  or `PA_DECODE:` log line from `pa_decode.py`'s `_LOGGER.info`.

## Pitfalls & anti-patterns
- This kernel **is** the `ROCM_ATTN` fallback cliff — if a profile shows it running when you expected asm,
  an AITER/HIP path is missing for your head size (2.7–4.4× slower).
- `num_stages>1` hurts the fused decode kernel; keep at 1.
- Don't expect to beat aiter asm decode; the win is portability/editability.
- per-token vs per-tensor fp8 scale shapes differ — passing the wrong rank silently picks the wrong path.

## Worked example
A model with head_dim=144 (not in aiter's supported {64,80,96,112,120,128,192,256}):
1. aiter custom path can't run → it falls back to **this** Triton decode kernel (the cliff).
2. Profile shows `_paged_attn_decode_*` Triton kernel, ~3× slower than the served TPOT target.
3. Options: pad head to 192/256 to re-enter the asm path, or accept Triton + tune `waves_per_eu`/`BLOCK_N`.
4. For ctx 16K: `max_num_partitions = ceil(16384/1024) = 16` → V2 path (since seqlen>8192).

## How to verify (bench + oracle)
`AMDGCN_ENABLE_DUMP=1` → `global_load_dwordx4` on KV reads, `v_mfma_*16x16`, no `scratch_`. Isolated
decode bench vs aiter/HIP at the served shape; greedy temp=0 parity. Gate: confirm it's the *intended*
backend (not an accidental cliff) AND parity.

## Alternatives / cross-links
./aiter.md (asm decode) · ./ck.md · ../../attention_prefill_fmha/backends/triton.md ·
../../mla_attention/backends/triton.md · `triton_levers/` ·
../../chunked_prefill/overview.md · ../overview.md.

## Sources
- aiter Triton paged-decode (on-box `ROCm/aiter@a6bb499375849eec45d68c5ccaebc8865fd422c0:aiter/ops/triton/attention/pa_decode.py`: `_SEQ_PARTITION_SIZE=1024`, V1≤8192 dispatch, per-token/per-tensor fp8 scale).
- Triton AMD knobs (num_stages=1, wave64): https://github.com/triton-lang/triton/blob/main/third_party/amd/backend/compiler.py
- ROCM_ATTN Triton fallback cliff (2.7–4.4× slower): https://vllm.ai/blog/2026-02-27-rocm-attention-backend
