---
title: Dense BF16 gated GEMM on AITER
kind: sota_card
operator: gemm_a16w16_gated
backend: aiter
gens: [gfx942, gfx950]
dtypes: [bf16, fp16]
regimes: [decode]
status: available
---

# Dense gated GEMM on AITER

`gemm_a16w16_gated` fuses a dense gate/up projection with the elementwise gated
activation. It accepts `x[M,K]` and packed `weight[2N,K]`, computes both projection
halves, applies an optional activation such as SiLU to the gate half, multiplies it
by the up half, and writes only `output[M,N]`.

## Fusion boundary

Use it for an observed contiguous chain such as:

```text
gate/up GEMM -> SiLU-and-multiply -> downstream projection
```

This is a larger boundary than replacing an existing standalone
`silu_and_mul` kernel. The fused GEMM removes the activation launch and avoids
writing and rereading the packed `[M,2N]` intermediate tensor.

## API

```python
from aiter.ops.triton.gemm.basic.gemm_a16w16_gated import gemm_a16w16_gated

output = gemm_a16w16_gated(
    x,
    packed_gate_up_weight,
    dtype=x.dtype,
    activation="silu",
)
```

## Constraints

1. The packed output dimension must be even.
2. Gate and up weights must occupy the first and second halves consistently.
3. Benchmark the exact `(M, N, K)` shape against the production GEMM backend.
4. Validate activation numerics; Triton SiLU and framework SiLU can differ slightly.
5. Do not infer that the boundary is already satisfied merely because a standalone
   `SiluAndMul` operator exists.

## Larger fusion warning

`ff_a16w16_fused_gated` can fuse gate/up, activation, and down projection, but it
uses atomic accumulation and requires a zeroed output buffer. It must win an exact
shape benchmark before integration; a wider fusion boundary is not automatically
faster.

## Source

`aiter/ops/triton/gemm/basic/gemm_a16w16_gated.py`
