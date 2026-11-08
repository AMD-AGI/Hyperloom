# SGLang ROCm Optimization Techniques — Distilled from Production PRs

These are core patterns for optimizing an MXFP4 DSA/MLA model on MI355X with SGLang.
Each section describes the **general idea** — look for similar opportunities in new models.

---

## Pattern 1: Silently-disabled features (check first, always)

**Core idea:** Many ROCm/AMD-specific optimizations exist in the code but are gated on `is_cuda()` or similar conditions, meaning they silently fall back to slower paths on HIP without any error.

**What to look for:**
- Any `if is_cuda():` or `if not _is_hip:` guard around a kernel dispatch
- Flags that say they're enabled but never actually set the enabling variable (log says "enabled" but the bool is never flipped)
- Features documented in server_args that have no effect on ROCm

**Examples:**
- AllReduce fusion claimed "enabled" in logs but the flag was commented out
- EAGLE verify rejection sampling ran on the CUDA path but fell back to greedy on HIP → a correctness bug, not only a speed one
- DSA CUDA graph capture skipped target-verify passes on HIP → entire EAGLE spec-decode ran eager
- Aiter top-k backend was registered but never selected as default

**How to find these:** Search for `is_cuda()`, `is_hip()`, `_is_cuda`, `_is_hip` in the SGLang source. Any branch that differs between CUDA and HIP is a candidate. Check if the HIP branch does something slower or is a no-op.

---

## Pattern 2: Kernel fusion — eliminate redundant launches

**Core idea:** GPU launch overhead accumulates. If two kernels write to the same tensor or read overlapping inputs, they can often be fused into one. The gain is especially large at small batch sizes (decode, not prefill) where compute time is short relative to launch time.

**What to look for:**
- Sequential operations on the same tensor (write bf16, then cast to fp8 → fuse)
- Separate kernel for a reduction that could be a histogram inside the main kernel
- Two-pass decode chains where pass 1 produces intermediate that pass 2 immediately reads

**Examples:**
- Two-pass fp8 Q construction (write bf16 → cast to fp8) → one fused kernel
- 12-kernel DSA indexer decode chain → 4 fused kernels with histogram inside
- Absorb BMM + rope + KV cache write as separate passes → fused `fused_fp8_bmm_rope_cat_and_cache_mla`
- Top-k scoring + page-table transform as separate ops → single fused kernel

**Where to look:** a decode profile — any two adjacent short ops are candidates. The indexer path (attn.indexer) is especially launch-heavy.

---

## Pattern 3: Shape compatibility — stay on the fast kernel path

**Core idea:** Fast kernels have hard shape constraints (head count must be 16, tile must be 64). When TP sharding produces an unsupported shape, code silently falls back to a slower path. The fix is often just padding/unpadding — trivial code, large gain.

**What to look for:**
- At TP=8, a 64-head model gives 8 q-heads per rank. Many kernels require 16.
- Tile size constraints (FlyDSL, tilelang often need power-of-2 heads)
- Any fallback to `tilelang` or `torch` when `aiter` is expected

**Examples:**
- TP=8 → 8 q-heads → FlyDSL requires 16 → pad to 16, discard extra half, instead of the tilelang fallback
- Same padding needed for prefill path separately from decode path

**How to find:** Look for `assert nhead in [16, 32, ...]` or similar in kernel dispatch code. Check what `self.need_pad_heads` triggers in `dsa_backend.py`. For a new model, count heads-per-rank at your TP setting.

---

## Pattern 4: Decode-vs-prefill path divergence

**Core idea:** A kernel optimized for decode (few tokens, long KV) can hurt prefill (many tokens, short KV). Features must be gated on forward mode, not just on model type.

**What to look for:**
- A single flag that enables a kernel for "all forwards" when it should only apply to decode
- ForwardBatch.forward_mode checks — `is_decode_or_idle()` vs `is_extend()` vs `is_prefill()`

**Examples:**
- Rope+KV fusion optimized for decode shape, but enabled for all forwards → prefill regression
  Fix: gate on `forward_batch.forward_mode.is_decode_or_idle()`.

**How to find:** Profile decode and prefill separately. If a "fix" helps decode but hurts prefill (or vice versa), check whether it's properly mode-gated.

---

## Pattern 5: Redundant work elimination

**Core idea:** If an operation is mathematically a no-op for your checkpoint (e.g., multiply by scale=1.0 for a bf16 model), but the code does it anyway — remove it.

**What to look for:**
- Dequantization of weights that are already bf16 (scale=1.0 is a no-op)
- Transpose or reshape that produces an identical tensor (duplicate materialization)
- Softmax at topk=1 (can't change argmax — always skip)

**Examples:**
- Per-forward weight dequantization in absorb BMMs is a no-op for a bf16 checkpoint → remove it
- Softmax at topk=1 removed (argmax is invariant to monotone transforms)

**How to find:** Read the kernel entry points in `dsa_backend.py` and `mla.py`. Check if scale factors are actually != 1.0 for your model. Profile with rocprofv3 to find kernels with <5% arithmetic utilization.

---

## Pattern 6: Sampling kernel bottlenecks at small batch

**Core idea:** Sampling kernels (top-p, renorm) are often written as one-block-per-row. At small batch (CONC=1, 6 rows with EAGLE), this leaves almost every CU idle. Restructure to radix-sort or split-row parallelism.

**What to look for:**
- `top_p_renorm`, `_top_p_renorm_kernel` in SGLang source
- Any kernel with a grid size that equals `batch_size` (one block per sample)
- High kernel time at CONC=1 that improves with CONC=32 (batch-bound, not compute-bound)

**Examples:**
- top-p renorm: pivot-search one-block-per-row → split-row radix, for the few rows of an EAGLE draft

**How to find:** Compare decode time per token at CONC=1 vs CONC=16. If it doesn't scale linearly, look at sampling/scheduling kernels.

---

## Pattern 7: MTP / EAGLE — dtype and metadata mismatches on HIP

**Core idea:** EAGLE speculative decoding has a verify pass (target model checks draft tokens). This path is less exercised on HIP, so dtype mismatches and metadata sizing bugs are more likely there.

**Specific bugs that apply to any DSA model with EAGLE:**
- `_forward_aiter` initialized metadata with `q_dtype=bfloat16` at model load, but runtime Q is fp8 → persistent kernel aborts in EAGLE draft steps
- `page_table_1` is `repeat_interleave`d by `speculative_num_draft_tokens` during target_verify → `num_rows != bs` → kv_indptr out of bounds
- Output tensor inherits Q dtype (fp8) but AITER reduce kernel requires bf16 output
- ASM kernel requires both `q_scale` and `kv_scale` when Q is fp8 — neither set if kv_cache is fp8

**Fixes (all in `dsa_backend.py`):**
1. `num_rows = page_table_1.shape[0]` — use actual rows, not batch_size
2. `q_scale = kv_scale = torch.ones(...)` when `kv_cache.dtype == fp8_dtype`
3. `o_dtype = torch.bfloat16` — force bf16 output regardless of Q dtype
4. Disable persistent mode — metadata sized at init with wrong q_dtype

**These 4 fixes are required for any DSA model using --dsa-decode-backend aiter + EAGLE on ROCm.**

---

## Where to look in a new model

When starting a campaign on a new DSA/MLA model:

1. **Check Pattern 1 first** — run grep for silently-disabled features. Little reading, potentially a large gain.
2. **Profile the decode step** — identify the top-3 ops. Don't guess.
3. **Check Pattern 3** — count heads-per-rank at your TP setting. If not a power of 2 that the kernel expects, padding is free gain.
4. **Check Pattern 4** — make sure any new kernel is mode-gated (decode only unless proven safe for prefill).
5. **Apply Pattern 7 patches** if using `--dsa-decode-backend aiter` + EAGLE — they're required, not optional.
6. **Profile sampling path at CONC=1** (Pattern 6) if EAGLE acceptance rate matters.

## DSA indexer

The DSA indexer (`attn.indexer`) runs on a subset of the layers.
A 4-kernel fused version exists (Pattern 2). Check if it's enabled on your model:
```bash
docker exec $CONTAINER grep -r "fused.*indexer\|four.kernel\|dsa_fused" /sgl-workspace/ --include="*.py" | head
```
If not enabled, `--enable-dsa-fused-indexer` flag may be available (gfx950 only).
