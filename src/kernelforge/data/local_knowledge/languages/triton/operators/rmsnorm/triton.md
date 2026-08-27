---
title: rmsnorm on triton — SOTA card
kind: sota_card
operator: rmsnorm
backend: triton
gens: [gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz]
regimes: [prefill, decode, both]
status: sota
updated: 2026-06-08
sources:
  - ROCm/aiter@a6bb4993:aiter/ops/triton/normalization/rmsnorm.py
  - ROCm/aiter@a6bb4993:aiter/ops/triton/_triton_kernels/normalization/rmsnorm.py
  - https://rocm.docs.amd.com/en/latest/how-to/llm-fine-tuning-optimization/optimizing-triton-kernel.html
---

# rmsnorm × triton

## TL;DR
Triton is the **authorable SOTA** for RMSNorm on MI300X — aiter *itself* ships a Triton impl as its
portable tier, and it is competitive with asm/CK because the op is **bandwidth-bound** (the compiler can't
do worse than streaming the row through HBM once each way). Reach for Triton when you need a **fused variant
the library lacks**, a new/odd shape, or `torch.compile` codegen. The reference body is aiter's
`_triton_kernels/normalization/rmsnorm.py`: a **persistent grid** + a **two-pass blocked** fallback for
rows too wide to hold in registers. The lever on AMD is keeping `num_warps` low (memory-bound), 128-bit
loads, and fp32 accumulate — not block-size sweeps.

## SOTA implementation(s)
| impl | source (`repo@commit:path`) | gens / dtypes | measured perf | when it's best |
|---|---|---|---|---|
| aiter Triton `_rms_norm_kernel` | `aiter/ops/triton/_triton_kernels/normalization/rmsnorm.py` | gfx942/950, bf16/fp16 | bandwidth-bound; persistent `min(rows,num_sms)` grid, `.cg` loads, `num_stages=2` blocked pipeline | the library Triton tier / fused variants |
| `_fused_add_rmsnorm_kernel`, `_quant_rms_norm_kernel`, `_quant_fused_add_rmsnorm_kernel` | same file | gfx942/950, bf16/fp8 out | one read + one write incl. residual / quant | fused_add_rmsnorm / fused_norm_quant |
| `_rmsnorm_kernel_large_m_small_n` | same (`_should_use_large_m_small_n`: `M>8192 & N≤2048`) | gfx942/950 | 2D-blocked, `BLOCK_N=next_pow2(N)`, `BLOCK_M=clamp(16384//BLOCK_N, 8, 32)`, `num_warps=8` | tall-skinny norm (many short rows) |
| `_rmsnorm_bwd_triton` (+ `_rmsnorm_bwd_dg_reduce_triton`) | same | gfx942/950 | training: per-row dx + 2-stage dγ reduce | RMSNorm backward (training) |
| Triton tutorial 05-layer-norm (RMS variant) | https://triton-lang.org/main/getting-started/tutorials/05-layer-norm.html | generic | row-per-program | learning / from-scratch reference |

### What the SOTA kernel actually does (on-box Triton)
The forward picks **single-pass** (whole row fits) vs **two-pass blocked** by `use_blocked(x)`, and runs a
**persistent loop** so each of `min(rows, num_sms)` programs sweeps strided rows:

```python
# ROCm/aiter@a6bb4993:aiter/ops/triton/normalization/rmsnorm.py
def num_programs(x): return min(x.shape[0], get_num_sms())
def block_size(x):   return min(65536 // x.element_size(), triton.next_power_of_2(x.shape[1]))
def use_blocked(x):  return x.shape[1] > block_size(x)
```
```python
# ...the kernel body (_rms_norm_kernel), single-pass branch:
for row_idx in tl.range(row_start, n_rows, NUM_PRGMS, num_stages=2):  # persistent, software-pipelined
    input_ptrs = tl.multiple_of(input_ptr + row_idx*stride + col_offsets, (16,))   # 16B-aligned ⇒ dwordx4
    row = tl.load(input_ptrs, mask=mask, other=0.0, cache_modifier=".cg").to(tl.float32)  # read-once .cg
    g   = tl.load(g_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)
    norm_factor = tl.math.rsqrt(tl.sum(row*row, axis=-1)/n_cols + epsilon)   # fp32 reduction
    tl.store(out_ptrs, (row*norm_factor*g).to(output_ptr.type.element_ty), mask=mask)
```
The **blocked** branch (taken when `N > 65536/elt`, e.g. N>32768 bf16) splits into an explicit
`Σx²` accumulation pass over `tl.cdiv(N,BLOCK)-1` blocks with `num_stages=2`, then a normalize+store pass —
this is a memory-pipelining choice (can't hold the whole row in registers), not a numerics one. `.cg`
(cache-global, bypass-L2-on-evict) marks the input as streamed read-once.

## Config space / knobs
| knob | recommended | why |
|---|---|---|
| `num_warps` | **2–4** (start 2) | memory-bound; 8 (NVIDIA default) spills VGPRs → 3–5× slower |
| `block_size` | `min(65536//elt, next_pow2(N))` (auto) | single-pass boundary; do not hand-raise past 65536/elt |
| `use_blocked` | `N > block_size` (auto) | two-pass guard; **keep it** when authoring or huge N crashes/cliffs |
| `NUM_PRGMS` | `min(rows, get_num_sms())` (auto) | persistent grid; each program strides rows by `NUM_PRGMS` |
| `num_stages` | 2 (blocked inner loop) | software-pipeline the block loads/stores |
| `cache_modifier` | `.cg` on x | read-once; don't pollute L2 |
| `tl.multiple_of(ptr, 16)` | always | promises 16-B alignment ⇒ `global_load_dwordx4` |
| accumulate | `x.to(tl.float32)` before `x*x` | fp32 `Σx²` |
| `waves_per_eu` | 3–4 | VGPR-light ⇒ push occupancy |

Verify ISA: `AMDGCN_ENABLE_DUMP=1 ... | grep -c global_load_dwordx4` should be nonzero (128-bit loads).

## Numerics / parity
- **fp32 `Σx²`**, fp32 γ promote, ε inside the mean (`rsqrt(Σx²/N + ε)`) — matches the asm/CK contract.
- **One-pass vs two-pass vs Welford**: RMSNorm needs only `Σx²` (no mean subtraction), so a single fp32
  accumulator is accurate to LLM hidden dims; the **two-pass blocked** path exists purely to fit wide rows
  in registers, **not** for stability, and it produces an identical result up to fp32 rounding. There is no
  Welford here (Welford trades extra fp32 ops for streaming variance — unnecessary when you don't subtract
  a mean).
- **Reduction order differs** from asm/CK (persistent strided sweep vs block-reduce) → an argmax tie can
  flip → **greedy re-gate** on any backend swap.
- **fp8 fused-quant output**: `_quant_rms_norm_kernel` computes the row in fp32, finds `row_max`, writes the
  per-token scale via `_per_token_quant` (`scale = row_max/DTYPE_MAX`), then casts to fp8 — fnuz on gfx942.
  In the blocked path the fp32 row is staged in an `aux` fp32 scratch tensor so the quant pass reads back
  the exact pre-quant values. See [numerics.md](numerics.md) and fused_norm_quant.

## Integration (rebind seam)
- **Direct**: `from aiter.ops.triton.normalization.rmsnorm import rms_norm` (or `rmsnorm2d_fwd_with_add`,
  `rmsnorm2d_fwd_with_dynamicquant`, `_with_smoothquant`).
- **`torch.compile`**: Inductor emits a Triton RMSNorm for `nn.RMSNorm` / the decomposed pattern under
  `max-autotune`; wire the AMD knobs (`waves_per_eu`, `num_warps`) via `torch._inductor.config`.
- **SGLang/vLLM**: use this as the fallback when aiter asm/CK has no tune for a shape.
- **Verify**: `TRITON_PRINT_AUTOTUNING=1` shows the winning config; a Triton-mangled `_rms_norm_kernel`
  name in `rocprofv3` confirms the Triton tier is actually running.

## Pitfalls & anti-patterns
- ⚠ `num_warps=8` carried over from an NVIDIA kernel → VGPR spill, 3–5× slower. Start at 2–4.
- ⚠ Removing the `use_blocked` two-pass guard and forcing single-pass for huge N (>32768 bf16) → register
  blow-up / perf cliff / OOM-of-registers. Keep the guard.
- ⚠ Forgetting `x.to(tl.float32)` before squaring → silent drift at N≥4096 (bf16 `Σx²` loses bits fast).
- ⚠ Dropping `tl.multiple_of(ptr,16)` → compiler falls back to `dwordx1` loads, ~4× the load instructions.
- ⚠ Authoring a quant variant without the `aux` fp32 scratch in the blocked path → you'd re-derive the row
  from re-loaded inputs and lose bit-exactness vs the single-pass quant.

## How to verify (bench + oracle)
- `TRITON_PRINT_AUTOTUNING=1` for the winner; isolated bench vs aiter asm/CK at the model hidden dim
  (median of ≥3 warm); `AMDGCN_ENABLE_DUMP=1 | grep global_load_dwordx4` for 128-bit loads; greedy parity
  e2e against the asm/CK tier.

## Worked example (wide MoE hidden N=12288, MI300X, bf16)
`block_size = min(65536//2, next_pow2(12288)) = min(32768, 16384) = 16384`; `12288 ≤ 16384` ⇒ **single-pass**
(the whole row fits). Grid = `min(rows, ~304 SMs)`; with 4096 rows you launch 304 persistent programs, each
sweeping ~13 rows. Each row is 12288·2 = 24 KiB read + 24 KiB write; `tl.multiple_of(...,16)` ⇒ 12288/8 =
1536 `dwordx4` loads. Now take **N=32768** (a very wide norm): `block_size = min(32768, 32768) = 32768`, still
`N ≤ block_size` ⇒ single-pass; bump to **N=40960** and `use_blocked` flips true → the kernel runs the
two-pass `Σx²`-then-normalize loops with `num_stages=2`. Set `num_warps=2`, `waves_per_eu=3`, confirm the
ISA dump shows `dwordx4`, and the kernel should sit near the HBM bandwidth floor.

## Alternatives / cross-links
aiter (asm/CK tiers + dispatch) · hip · vllm_kernels ·
triton_levers/patterns §5 (fused softmax/RMSNorm template) · [tuning.md](tuning.md) ·
optimization/vectorization_and_coalescing · optimization/occupancy_and_registers.

## Sources
- aiter Triton rmsnorm wrappers (persistent grid, `block_size`/`use_blocked`, large-m-small-n, backward):
  `ROCm/aiter@a6bb4993:aiter/ops/triton/normalization/rmsnorm.py`.
- aiter Triton kernel bodies (single/two-pass, `.cg`, `num_stages=2`, `_per_token_quant`, `aux` scratch):
  `ROCm/aiter@a6bb4993:aiter/ops/triton/_triton_kernels/normalization/rmsnorm.py`.
- Memory-bound AMD Triton knobs (num_warps, next_pow2 block, 128-bit loads):
  https://rocm.docs.amd.com/en/latest/how-to/llm-fine-tuning-optimization/optimizing-triton-kernel.html.
- Triton layer-norm tutorial (persistent / two-pass reference):
  https://triton-lang.org/main/getting-started/tutorials/05-layer-norm.html.
