---
title: Triton on AMD — starting kernel templates
kind: language
lever: triton_templates
gens: [gfx950]
updated: 2026-08-28
sources:
  - https://rocm.docs.amd.com/en/latest/how-to/llm-fine-tuning-optimization/optimizing-triton-kernel.html
  - https://rocm.docs.amd.com/projects/ai-developer-hub/en/latest/notebooks/gpu_dev_optimize/triton_kernel_dev.html
  - https://github.com/sgl-project/sglang/pull/2601
---

# Kernel templates

Starting bodies that are **already CDNA-tuned**: wave64-aware, `num_warps` low, ≥1024 programs,
MFMA-16, `OPTIMIZE_EPILOGUE=1`. Copy one, then tune with `triton_knob_space.md`.

## Route here when
You know what you are writing (GEMM / attention / reduction) and want a correct starting point instead
of porting an NVIDIA kernel and discovering the deltas one by one.

## 1. Dense GEMM — L2-swizzled, MFMA-16

```python
import torch, triton, triton.language as tl

def _amd_configs():
    cfgs = []
    for BM, BN, BK in [(128,128,64), (128,256,64), (256,128,64), (128,64,64)]:
        for nw in (4, 8):                        # wave64: 4 warps = 256 threads
            for we in (2, 3):
                cfgs.append(triton.Config(
                    {"BLOCK_M":BM, "BLOCK_N":BN, "BLOCK_K":BK,
                     "GROUP_SIZE_M":8,            # = XCD count, for L2 reuse
                     "matrix_instr_nonkdim":16,   # 4 C-regs/lane vs 32x32's 16
                     "waves_per_eu":we},
                    num_warps=nw, num_stages=2))  # AMD: 2 for a single GEMM
    return cfgs

@triton.autotune(configs=_amd_configs(), key=["M","N","K"])
@triton.jit
def gemm(a_ptr, b_ptr, c_ptr, M, N, K,
         sam, sak, sbk, sbn, scm, scn,
         BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
         GROUP_SIZE_M: tl.constexpr):
    pid = tl.program_id(0)
    npm = tl.cdiv(M, BLOCK_M); npn = tl.cdiv(N, BLOCK_N)
    # L2-friendly swizzle: group rows so neighbours reuse B out of the same XCD's L2
    nig   = GROUP_SIZE_M * npn
    gid   = pid // nig
    fpm   = gid * GROUP_SIZE_M
    gsm   = min(npm - fpm, GROUP_SIZE_M)
    pid_m = fpm + ((pid % nig) % gsm)
    pid_n = (pid % nig) // gsm

    offs_m = (pid_m*BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_n = (pid_n*BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + offs_m[:,None]*sam + offs_k[None,:]*sak
    b_ptrs = b_ptr + offs_k[:,None]*sbk + offs_n[None,:]*sbn

    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)          # -> AGPR accumulator
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        km = offs_k[None,:] < K - k*BLOCK_K
        a = tl.load(a_ptrs, mask=km, other=0.0)             # -> global_load_dwordx4
        b = tl.load(b_ptrs, mask=offs_k[:,None] < K - k*BLOCK_K, other=0.0)
        acc = tl.dot(a, b, acc)                             # -> ds_read_b128 + v_mfma
        a_ptrs += BLOCK_K*sak; b_ptrs += BLOCK_K*sbk

    c = acc.to(c_ptr.dtype.element_ty)                      # OPTIMIZE_EPILOGUE=1 drops the convert
    ocm = pid_m*BLOCK_M + tl.arange(0, BLOCK_M)
    ocn = pid_n*BLOCK_N + tl.arange(0, BLOCK_N)
    tl.store(c_ptr + scm*ocm[:,None] + scn*ocn[None,:], c,
             mask=(ocm[:,None] < M) & (ocn[None,:] < N))
```

Two things that are easy to miss:
- **`GROUP_SIZE_M=8`** aligns block grouping to the **8 XCDs**. L2 is per-XCD, so this is what makes
  the B-panel reuse actually hit cache.
- **Pad the leading dimension off a 512 B multiple** for the TN layout — if `K % 256 == 0`, allocate
  `lda = ldb = K + 128`. Otherwise you can hit the L2 tag-RAM cliff.

## 2. Skinny / decode GEMM with SPLIT_K

Decode shapes (M = 1..64, large K) produce only a handful of output tiles, so most of the 256 CUs sit
idle. `SPLIT_K` splits the K reduction across programs that `tl.atomic_add` into C.

```python
configs = [triton.Config(
    {"BLOCK_M":64, "BLOCK_N":128, "BLOCK_K":64, "GROUP_SIZE_M":8, "SPLIT_K":sk,
     "matrix_instr_nonkdim":16, "waves_per_eu":3},
    num_warps=4, num_stages=2) for sk in (1, 2, 4, 8, 16)]
# in the kernel: pid_k = tl.program_id(1); loop k over [pid_k*step, ...];
#                zero-init C; tl.atomic_add(c_ptrs, c, mask)
```

Costs a C zero-init plus atomics. Skip when M·N already yields ≥1024 tiles.
**This is the one regime where authored Triton reliably beats hipBLASLt.**

## 3. fp8 GEMM — get the dialect right

**On gfx950 use OCP fp8.** The FNUZ types below are the **gfx942** path and are the classic porting
trap in both directions:

| Target | dtype | Note |
|---|---|---|
| **gfx950** | OCP `float8e4nv` / `float8e5` | also MXFP block-scaled via `mfma_scale_*_f8f6f4` |
| gfx942 | `tl.float8e4b8` (E4M3 fnuz) / `tl.float8e5b16` | OCP `float8_e4m3fn` raises `Unsupported conversion 'f8E4M3FN'` |

```python
a = a.to(tl.float8e4nv)          # gfx950: OCP
b = b.to(tl.float8e4nv)
acc += tl.dot(a, b) * a_scale[:, None] * b_scale[None, :]   # block-scaled
```

SGLang/vLLM call `normalize_e4m3fn_to_e4m3fnuz` before the matmul **when targeting gfx942**
(sglang PR #2601) — that conversion is exactly what you must *not* do on gfx950.

`supported_fp8_dtypes` on the AMD backend = `("fp8e4nv", "fp8e5", "fp8e5b16", "fp8e4b8")`; the FNUZ
MFMA types are `fp8e4b8` / `fp8e5b16`.

## 4. Flash-Attention shape

The high-value fused kernel: two `tl.dot`s (QKᵀ then PV) plus online softmax.

AMD specifics:
- **`num_stages=1`** — two dots already saturate LDS and registers.
- **`num_warps=4`**.
- Optionally `schedule_hint="attention"` (or `"memory-bound-attention"` for decode).
- Keep the softmax reduce **wave64-full**: round the reduced dimension to a power of 2 ≥ the row width.
  A reduced dim < 64 wastes lanes.
- Verify dense `v_mfma` between the dots and no scratch spill.

## 5. Fused softmax / RMSNorm / SiLU (memory-bound)

```python
@triton.autotune(configs=[triton.Config({}, num_warps=nw) for nw in (2,4,8)], key=["n_cols"])
@triton.jit
def softmax(out_ptr, in_ptr, isr, osr, n_cols, BLOCK_SIZE: tl.constexpr):
    row  = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    x    = tl.load(in_ptr + row*isr + cols, mask=cols < n_cols, other=-float("inf"))
    x    = x - tl.max(x, 0)                  # 64-lane wave max
    num  = tl.exp(x); den = tl.sum(num, 0)   # 64-lane wave sum
    tl.store(out_ptr + row*osr + cols, num/den, mask=cols < n_cols)
```

Memory-bound rules: `num_warps=2/4`, `num_stages=1`, and
**`BLOCK_SIZE = next_pow2(n_cols)`** so the wave reduce is full. Same shape for fused-add-RMSNorm and
SiLU·mul.

## 6. Grid sizing (universal)

Target **≥1024 programs** so the scheduler can hide latency across 8 XCDs / **256 CUs**. If a shape
cannot reach that (skinny GEMM), use `SPLIT_K`. Confirm with `TRITON_PRINT_AUTOTUNING=1`.

## Verify

Every template above should produce, in the inner loop: `global_load_dwordx4`, `ds_read_b128`, dense
`v_mfma_*`, no `v_accvgpr_*`, and `.private_segment_fixed_size: 0`. Check with
`triton_isa_check.md` — a template that compiles is not a template that is fast.

## Sources
- Optimizing Triton kernels (GEMM swizzle, `OPTIMIZE_EPILOGUE`, softmax): https://rocm.docs.amd.com/en/latest/how-to/llm-fine-tuning-optimization/optimizing-triton-kernel.html
- AI Developer Hub Triton kernel dev tutorial (templates, autotune): https://rocm.docs.amd.com/projects/ai-developer-hub/en/latest/notebooks/gpu_dev_optimize/triton_kernel_dev.html
- FNUZ fp8 normalization in `tl.dot`: https://github.com/sgl-project/sglang/pull/2601
- Grid sizing / SPLIT_K / Tagram: https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/workload.html
