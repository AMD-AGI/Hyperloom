---
title: rope — numerics & parity
kind: technique
operator: rope
gens: [gfx942, gfx950]
dtypes: [bf16, fp16, fp32]
regimes: [both]
updated: 2026-06-08
sources:
  - /sgl-workspace/aiter/aiter/ops/rope.py
  - https://github.com/vllm-project/vllm/blob/main/csrc/pos_encoding.cu
  - https://github.com/vllm-project/vllm/pull/22593
---

# rope — numerics & parity

RoPE is a rotation (no reduction) so it's numerically gentle, but **style and dimensioning bugs** are the
common failures.

## 1. NeoX vs GPT-J style — must match the checkpoint
- **NeoX** (`is_neox=True`): rotate-halves — pair dim `i` with `i + d/2`.
- **GPT-J** (`is_neox=False`): rotate adjacent pairs `(2i, 2i+1)`.
Same math, different pairing → applying the wrong one scrambles Q/K → garbage attention. The flag comes
from the model config; vLLM's `rotary_embedding(..., is_neox)` carries it.

## 2. cos/sin in fp32
Compute/store the cos/sin cache in **fp32**; the rotation multiply is fp32 then converts to bf16. bf16
cos/sin at large positions loses angle precision → drift over long context.

## 3. Partial rotation (rotary_dim < head_size)
GLM-4.1V and others rotate only the first `rotary_dim` dims; the rest pass through unchanged. A kernel that
assumes `rotary_dim == head_size` reads out of bounds → illegal memory access (vLLM #22593 fixed exactly
this for the MRoPE Triton kernel; #39625 is a related partial-rotary shape mismatch). Bound the rotation
loop by `rotary_dim`.

## 4. Scaling variants change the angle table, not the kernel
YaRN, Dynamic-NTK, Linear scaling alter `θ_i` (the base / frequency) — they change the **cos_sin_cache**
contents, not the rotation kernel. Multiple LoRA scaling factors → multiple caches (vLLM batches them).
Verify the cache matches the model's scaling config.

## 5. Parity
RoPE is exact up to fp rounding; the rotation order doesn't vary, so cross-backend parity is tight. After
swapping the RoPE kernel, greedy parity should be token-identical (unlike reductions). A divergence usually
means a **style/partial/scaling mismatch**, not rounding.

## Parity gate
1. isolated vs fp64 reference rotation: tight rel-err (no reduction → ~1e-3 bf16).
2. confirm `is_neox` matches config; `rotary_dim` bound correct; scaling cache correct.
3. greedy e2e: token-identical (RoPE is deterministic) — divergence = a style/dim bug.

## Sources
- aiter rope variants (is_neox, partial, cached): `/sgl-workspace/aiter/aiter/ops/rope.py`.
- is_neox / cos_sin_cache: https://github.com/vllm-project/vllm/blob/main/csrc/pos_encoding.cu.
- partial-rotary fix (rotary_dim < head_size): https://github.com/vllm-project/vllm/pull/22593, https://github.com/vllm-project/vllm/issues/39625.
