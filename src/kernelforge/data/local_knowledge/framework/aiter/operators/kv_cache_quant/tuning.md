---
title: kv_cache_quant — tuning
kind: operator_overview
operator: kv_cache_quant
gens: [gfx942, gfx950]
dtypes: [fp8_e4m3_fnuz, fp8_e4m3, int8]
regimes: [decode, prefill, both]
updated: 2026-07-14
sources:
  - vllm-project/vllm@HEAD:csrc/cache_kernels.cu
  - ROCm/aiter@b467ce342:aiter/ops/fused_qk_norm_rope_cache_quant.py
  - https://github.com/vllm-project/vllm/blob/main/vllm/envs.py
---

# kv_cache_quant — tuning

> KV quant is **not a throughput-tuning op** — the store is tiny (one token in decode) and the read is part
> of attention. The "tuning" is (1) **enabling** FP8 KV to get 2× capacity, (2) picking the **scale
> granularity** for accuracy, and (3) ensuring the **fused** write path is used so there is no separate KV
> quant launch. The system-level win comes from the bigger batch/context the freed HBM allows.

## The levers
1. **Enable**: `--kv-cache-dtype fp8` (or `fp8_e4m3` / `fp8_e5m2` / `int8`). This is the lever that
   matters — 2× KV capacity → larger max batch / context.
2. **Scale granularity**: per-tensor static (default, cheapest) → block/per-token (aiter `*_block_quant`)
   if accuracy needs it (numerics.md).
3. **Fuse the write**: use `fused_qk_norm_rope_cache_quant*` (aiter) or `reshape_and_cache_flash` (vLLM) so
   the quant rides the cache write — no separate launch. Standalone KV quant is a wasted pass.
4. **KV layout shuffle**: `VLLM_ROCM_SHUFFLE_KV_CACHE_LAYOUT=1` for the ROCM_AITER_FA backend at
   concurrency **≥32** (improves the paged read coalescing); must match between write and read.

## Decode vs prefill
- **decode**: the store is 1 token; cost is the launch — always fused into the QK-norm+RoPE+write chain.
  The hot path is the *read* (paged attention dequant), tuned in operators/attention_decode_paged.
- **prefill**: the store is the whole prompt; `reshape_and_cache` writes all slots; vectorize the
  scaled_convert.

## Capacity math (why it's worth it)
KV bytes = `2 · num_layers · num_kv_heads · head_dim · seq · batch · dtype_bytes`. FP8 halves
`dtype_bytes` (2→1) → 2× the `seq·batch` product fits in the same HBM. For long-context decode this is the
single biggest memory lever, often unlocking a batch size that more than pays back the small quant cost.

## Knobs (HIP cache-write kernel)
- vectorized `scaled_convert` over the head_dim; one thread block per token/slot.
- `block_size` (paged block), `x` (K-layout split) — match the cache allocation.
- `kv_cache_dtype` string selects the convert template (`DISPATCH_BY_KV_CACHE_DTYPE`).

## Sources
- vLLM reshape_and_cache store + dispatch: `vllm-project/vllm@HEAD:csrc/cache_kernels.cu`.
- aiter fused QK-norm+RoPE+KV-quant (block_size, slot_mapping, scales): `ROCm/aiter@b467ce342:aiter/ops/fused_qk_norm_rope_cache_quant.py`.
- KV shuffle / kv-cache-dtype env: https://github.com/vllm-project/vllm/blob/main/vllm/envs.py
