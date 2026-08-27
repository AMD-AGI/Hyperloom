---
title: sparse_attention_nsa — tuning
kind: technique
operator: sparse_attention_nsa
gens: [gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz]
regimes: [prefill, decode]
updated: 2026-08-23
sources:
  - https://github.com/fla-org/native-sparse-attention
  - https://arxiv.org/html/2508.18224v1
  - https://fergusfinn.com/blog/deepseek-v4-flash-mi300x/
---

# sparse_attention_nsa — tuning

## The three cost centers (tune in this order)
1. **Indexer / top-k selection** — scoring KV blocks and picking top-k. On DeepSeek sparse MLA this is
   the **fp8 MQA logits** kernel (`pa_mqa_logits` / `fp8_mqa_logits`); it dominates because it touches
   all (or compressed) KV. Online top-k (FLA) avoids materializing the score matrix.
2. **Gather + ragged attention** — the selected branch reads a **variable, non-contiguous** set of KV
   blocks per query. Flash Sparse Attention (FSA) shows the key wins: an **early-return** on empty index
   tensors, **decoupling** online-softmax statistics into a pre-compute kernel, and **decoupling**
   attention accumulation into a separate kernel — to avoid loop-order stalls from non-contiguous query
   access.
   **Before pricing this branch by how many blocks it reads, price it by whether its loop can be
   pipelined.** A `while` over a per-query index list, or a page pointer loaded inside the loop and
   dereferenced in the same iteration, silently forfeits `num_stages` pipelining *and* direct-to-LDS
   async copy. Iterating a shape-static superset (e.g. the contiguous causally-visible block range) and
   selecting with a mask does strictly more nominal work and can still be the faster kernel — measured
   ~2× on one gfx950 prefill kernel at 2–4× the block count. See
   `local_knowledge/common_methodology/optimization/loop_form_and_pipelining.md` for the signature, the
   rewrite, and the limits of that single data point.
3. **The matmul** — once gathered, it's a normal FA block; least of your worries.

## AMD Triton knobs (selected/compressed/sliding branches — triton_amd)
- `BLOCK_M / BLOCK_N` aligned to the NSA `block_size` (64). Keep `BLOCK_N == block_size` so a selected
  block is one KV tile (clean gather).
- `matrix_instr_nonkdim=16`, `num_warps=4` (avoid 8 spill), `num_stages=1` (fused FA body).
- `waves_per_eu=2–3`, `schedule_hint=attention`.
- `knobs.amd.use_buffer_ops=ON` — selected-block gather is a masked/indexed load; bounds-checked
  `buffer_load` matters.
- fp8 indexer: **fnuz** (`fp8e4b8`/`fp8e5b16`) on gfx942.

## Capture-safe metadata (the production lever on MI300X)
The DeepSeek-V4 bring-up's biggest single win was making the sparse-MLA-decode metadata **static,
capture-safe tensors** — no dynamic ragged allocations or host→device scalar writes under HIP-graph
capture. Rebuilding ragged index tensors each step (host sync) was the bottleneck; static buffers gave
**+8.6%** (2485→2699 tok/s/GPU). Tune the bookkeeping, not just the kernel.

## gfx942 dispatch routing
On CDNA3, route to aiter where a tuned path exists and **refuse dispatch (fall to Triton)** where the
AITER path is broken on gfx942 (AITER prefill MQA logits / sparse prefill logits were broken on gfx942).
The missing-entirely paths (paged MQA logits, sparse MLA prefill/decode) needed a ROCm helper →
Triton fallback. Generic Triton is "several × slower than a tuned kernel" — so verify which path runs.

## Autotune sketch
Sweep per branch keyed on `(seq, block_size, top_k, head_dim)`. Selected branch: `(BLOCK_M)∈{64,128}`,
`num_warps∈{4,8}`, `waves_per_eu∈{2,3}`. Indexer: tune the reduction tile + fp8 scale. Re-tune per
`top_k` and per context length.

## Verify the sparsity is real
Prefill time should scale with `top_k·block_size`, not `seq`. If it scales with `seq`, the selected
branch is reading all KV (gather/skip broken) or the indexer is the bottleneck — profile both. Caveat:
this test assumes the two candidate kernels pipeline equally well. A masked static-range kernel reads
more KV *by design* and will look "less sparse" on this test while being faster in wall clock — compare
times, and use the scaling test only to catch an unintended full-KV read.

## Sources
- FLA NSA (online top-k, block_size=64, fused selected+sliding): https://github.com/fla-org/native-sparse-attention
- FSA kernel optimizations (early-return, decoupled softmax/accumulate): https://arxiv.org/html/2508.18224v1
- MI300X capture-safe metadata +8.6%, gfx942 routing: https://fergusfinn.com/blog/deepseek-v4-flash-mi300x/
- aiter sparse MLA / mqa logits: `ROCm/aiter@a6bb49937:aiter/ops/triton/attention/{unified_attention_sparse_mla,pa_mqa_logits,fp8_mqa_logits}.py` (on-box).
