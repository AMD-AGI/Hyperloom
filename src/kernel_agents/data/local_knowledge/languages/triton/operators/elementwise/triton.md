---
title: elementwise on Triton — SOTA card
kind: sota_card
operator: elementwise
backend: triton
gens: [gfx942, gfx950]
dtypes: [fp32, bf16, fp16, fp8_e4m3_fnuz]
regimes: [both]
status: sota
updated: 2026-06-08
sources:
  - https://rocm.docs.amd.com/en/latest/how-to/llm-fine-tuning-optimization/optimizing-triton-kernel.html
  - https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/workload.html
  - https://github.com/triton-lang/triton/blob/main/third_party/amd/backend/compiler.py
---

# elementwise × triton

## TL;DR
Triton is the SOTA way to **author** a fused elementwise kernel on MI300X and is what Inductor emits.
For a *single* op it just matches the HIP bandwidth ceiling; its real value is **fusing a whole pointwise
expression into one kernel**. Tuning is trivial vs GEMM: pick `BLOCK_SIZE` a multiple of the 128-bit
vector width, `num_warps=2/4`, `num_stages=1`, and verify `global_load_dwordx4` in the ISA.

## SOTA implementation(s)
| impl | source | gens/dtypes | measured perf | when best |
|---|---|---|---|---|
| 1-D contiguous fused pointwise (grid-stride / large grid) | this card + [`../../../triton_levers/patterns.md`] §5 | gfx942/950, all | bandwidth-bound: **~3.5–4.3 TB/s** achievable (vs ~4.3 BabelStream peak) @ MI300X | any fused unary/binary/ternary chain |
| Inductor-generated pointwise kernel | [backends/pytorch_inductor.md](pytorch_inductor.md) | gfx942/950 | same ceiling, **auto-fused** | `torch.compile` graphs (default) |

```python
@triton.jit
def add_mul_clamp(a_ptr, b_ptr, out_ptr, s, lo, hi, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid*BLOCK + tl.arange(0, BLOCK)           # BLOCK%8==0 (bf16) -> dwordx4
    m = offs < n
    a = tl.load(a_ptr + offs, mask=m)                # contiguous -> global_load_dwordx4
    b = tl.load(b_ptr + offs, mask=m)
    o = tl.minimum(tl.maximum(a*s + b, lo), hi)      # whole chain in one pass, fp32 math
    tl.store(out_ptr + offs, o, mask=m)
# grid = (triton.cdiv(n, BLOCK),) with BLOCK=1024..4096 -> aim >=1024 programs
```

## Config space / knobs
- `BLOCK_SIZE`: 1024–8192, multiple of vec width (8 for bf16, 16 for fp8) so loads are 128-bit.
- `num_warps`: **2 or 4** (memory-bound); never 8 (VGPR spill, no benefit on a BW kernel).
- `num_stages`: **1** (no K-loop to pipeline).
- `waves_per_eu`: **3–4** to lift occupancy and hide HBM latency (knobs.md).
- `knobs.amd.use_buffer_ops=1` → `buffer_load/store` for cheap masked bounds-checking on the tail.
- Grid: enough programs for **≥1024** (workload guide); for huge tensors a grid-stride inner loop keeps
  programs bounded while still issuing `dwordx4` in the loop.

## Numerics / parity
bf16/fp16 auto-promote to fp32 in `tl.dot`-free arithmetic → matches torch. `tl.where` is NaN-safe.
fp8 cast: use `tl.float8e4b8` (fnuz) on gfx942. See [numerics.md](numerics.md).

## Integration (rebind seam)
Standalone: register as a torch custom op (`direct_register_custom_op`) so Inductor keeps it opaque and
fuses *around* it. In practice you usually let Inductor *generate* the pointwise kernel instead of
shipping one. The authored kernel matters when the fusion the compiler picks is suboptimal.

## Pitfalls & anti-patterns
- `num_warps=8` carried from NVIDIA → spill, slower on a BW kernel. Use 2/4.
- `BLOCK_SIZE` not a vec multiple → falls back to `dword`/`dwordx2`, ~2–4× fewer bytes/instr.
- Strided/broadcast operand kills 128-bit coalescing — materialize or restructure.
- Not verifying the ISA: `AMDGCN_ENABLE_DUMP=1` and grep `global_load_dwordx4`.

## How to verify
`TRITON_PRINT_AUTOTUNING=1`; achieved GB/s = `(read+write bytes)/median_time` vs ~4.3 TB/s; ISA shows
`_dwordx4`; atol parity vs `torch` eager.

## Alternatives / cross-links
hip (peak BW, full control) · pytorch_inductor (auto-fuse) ·
[tuning.md](tuning.md) · [fusion.md](fusion.md) ·
`../../../triton_levers/patterns.md` §5.

## Sources
- `global_load_dwordx4` in loops, `_b128`, ISA verify: https://rocm.docs.amd.com/en/latest/how-to/llm-fine-tuning-optimization/optimizing-triton-kernel.html
- ≥1024 grid, 16 B access, block sizing: https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/workload.html
- num_warps/num_stages/waves_per_eu/buffer_ops (HIPOptions): https://github.com/triton-lang/triton/blob/main/third_party/amd/backend/compiler.py
