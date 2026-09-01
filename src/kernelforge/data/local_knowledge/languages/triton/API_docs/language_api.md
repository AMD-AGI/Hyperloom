---
title: Triton language API (tl.*) — the kernel-body op reference
kind: api_reference
gens: [gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz, fp8_e5m2_fnuz, int8, fp32]
regimes: [both]
status: sota
updated: 2026-07-09
sources:
  - https://triton-lang.org/main/python-api/triton.language.html
  - https://triton-lang.org/main/getting-started/tutorials/index.html
---

# Triton language API (`tl.*`)

The device-body API you write inside a `@triton.jit` kernel. The Python surface is **identical on AMD and
NVIDIA** — what differs is lowering (see [../skills/optimize/triton_levers/triton_lowering.md](../skills/optimize/triton_levers/triton_lowering.md))
and the AMD dtype rules (FNUZ fp8). For tuning knobs see
[../skills/optimize/triton_levers/triton_knob_space.md](../skills/optimize/triton_levers/triton_knob_space.md); for the launch/
decorator surface see [programming_model.md](programming_model.md).

## Program / index
```python
pid  = tl.program_id(axis)          # this program's id along a grid axis (0/1/2)
n    = tl.num_programs(axis)
offs = tl.arange(0, BLOCK)          # BLOCK must be a compile-time power of 2
```

## Memory: load / store (with masks = predication)
```python
x = tl.load(ptr + offs, mask=offs < N, other=0.0)      # OOB lanes get `other`
tl.store(ptr + offs, x, mask=offs < N)
# Block pointers (structured tiling, cleaner masking / vectorization):
bp = tl.make_block_ptr(base, shape=(M,N), strides=(sm,sn),
                       offsets=(om,on), block_shape=(BM,BN), order=(1,0))
x  = tl.load(bp, boundary_check=(0,1))
bp = tl.advance(bp, (0, BK))
```
- A well-formed contiguous load lowers to `global_load_dwordx4`; masked tails want
  `knobs.amd.use_buffer_ops` for `buffer_load` HW bounds-check (see triton_levers/pitfalls).
- `cache_modifier` / `eviction_policy` args control L2 behavior.

## Compute
```python
acc = tl.dot(a, b, acc)                       # → v_mfma_*  (MFMA); acc is fp32
acc = tl.dot_scaled(a, a_s, "e4m3", b, b_s, "e4m3", acc=acc)   # block-scaled MXFP → see tl_dot_scaled_gfx950.md
y = tl.exp(x); y = tl.log(x); y = tl.sqrt(x); y = tl.sigmoid(x)   # tl.math.* elementwise
z = tl.where(cond, a, b)
z = tl.maximum(a, b); z = tl.minimum(a, b)
z = a * b + c                                  # standard operators fuse to FMA
```

## Reductions (round the reduced dim to ≥64 for a full wave64 reduce)
```python
m = tl.max(x, axis)     # row-max (softmax)
s = tl.sum(x, axis)     # row-sum
p = tl.cumsum(x, axis)  # scan
i = tl.argmax(x, axis)
```

## Atomics
```python
tl.atomic_add(ptr + offs, val, mask=...)      # split-K accumulate; also _max/_min/_cas/_xchg
```

## Dtypes & casts (AMD-specific)
```python
a = a.to(tl.float8e4b8)     # E4M3 FNUZ — the gfx942 MFMA fp8 (NOT tl.float8e4nv = OCP)
a = a.to(tl.float8e5b16)    # E5M2 FNUZ
c = acc.to(tl.bfloat16)     # epilogue downcast; OPTIMIZE_EPILOGUE=1 drops the convert_layout
```
`supported_fp8_dtypes` (AMD) = `fp8e4nv, fp8e5, fp8e5b16, fp8e4b8`; the **fnuz** MFMA types are
`fp8e4b8` / `fp8e5b16`. OCP `float8_e4m3fn` into `tl.dot` on gfx942 raises `Unsupported conversion`.

## Misc
```python
tl.cdiv(a, b)              # ceil-div (grid/tile math)
tl.static_assert(cond)    # compile-time check
tl.multiple_of / tl.max_contiguous   # alignment hints for vectorization
tl.debug_barrier()
```

## Sources
- Triton language reference (tl.load/store/dot/reduce/atomic/math, block pointers): https://triton-lang.org/main/python-api/triton.language.html
- Triton tutorials (softmax, matmul, flash-attention patterns): https://triton-lang.org/main/getting-started/tutorials/index.html
- AMD dtype/lowering specifics: [../skills/optimize/triton_levers/triton_lowering.md](../skills/optimize/triton_levers/triton_lowering.md), [../skills/optimize/triton_levers/triton_traps.md](../skills/optimize/triton_levers/triton_traps.md)
