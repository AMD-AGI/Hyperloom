---
title: softmax — tuning
kind: technique
operator: softmax
gens: [gfx942, gfx950]
dtypes: [bf16, fp16, fp32]
regimes: [both]
updated: 2026-06-08
sources:
  - /sgl-workspace/aiter/aiter/ops/triton/softmax.py
  - /sgl-workspace/aiter/aiter/ops/triton/_triton_kernels/softmax.py
  - https://rocm.docs.amd.com/en/latest/how-to/llm-fine-tuning-optimization/optimizing-triton-kernel.html
---

# softmax — tuning

Memory-bound row reduction with two reductions (max, sum). Same bandwidth playbook as the norms, plus the
**online single-pass** trick to avoid a second HBM read.

## 1. Online (single-pass) vs two-pass
aiter's `_softmax_kernel_online` streams the row in blocks, maintaining a running max `m` and running
sum, correcting the sum when the max grows:
```python
m_p = tl.maximum(m, m_p)              # new running max
row_sum = row_sum * tl.exp(m - m_p)  # correct previous partial sum
row_sum += tl.sum(tl.exp(row_block - m_p))
# second sweep: softmax_output = tl.exp(row_block - m) / row_sum
```
- **N fits a block**: load once, max + sum in registers, write — fully on-chip.
- **N > block (wide vocab)**: online keeps it **one HBM read** for the statistics; the normalize sweep
  re-reads (or recompute exp). Online avoids the 3-pass (max, sum, normalize) naive scheme.

## 2. Grid + wave64
- One program per row (routing/logits have many rows → fills the chip). For few rows, persistent
  `min(M, num_sms)`.
- aiter uses `num_warps = 8` for the online softmax (wide rows benefit from more lanes on the reduce);
  for narrow N (routing, N=experts ≤ 256) drop to 2–4.
- **Round the reduced dim to a power of 2** so the wave64 max/sum reduce is full — a reduced width < 64
  wastes lanes (ROCm Triton guidance).

## 3. Knob table
| knob | softmax setting | note |
|---|---|---|
| `num_warps` | 8 wide (vocab) / 2–4 narrow (routing) | aiter uses 8 online |
| `num_stages` | 1–2 | overlaps block loads |
| `BLOCK_SIZE` | next_pow2(N) (or block for wide N) | full wave reduce, 128-bit loads |
| grid | row-per-prog / `min(M,num_sms)` | fill 304 CUs |
| compute dtype | fp32 (exp + accumulate) | stability |

## 4. The real lever: don't run it standalone
For attention, softmax is **inside FMHA** (online/flash) — tuning belongs to the attention kernel
(`schedule_hint="attention"`, `num_stages=1`, two chained dots). For MoE routing, fuse softmax+topk
(aiter `topk_softmax`). Standalone tuning only matters for wide-vocab logits. See [fusion.md](fusion.md).

## Sources
- online softmax kernel (running max/sum + correction, num_warps=8): `/sgl-workspace/aiter/aiter/ops/triton/softmax.py`, `_triton_kernels/softmax.py`.
- power-of-2 reduced dim / wave64 full reduce: https://rocm.docs.amd.com/en/latest/how-to/llm-fine-tuning-optimization/optimizing-triton-kernel.html.
- attention schedule_hint / num_stages=1: perf_knowledge languages/triton_amd/patterns §4.
