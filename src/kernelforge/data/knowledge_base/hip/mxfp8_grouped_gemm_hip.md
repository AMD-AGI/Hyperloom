# HIP MX-FP8 Grouped GEMM — Know-Hows

Source: Primus-Turbo `feat/mxfp8-grouped-gemm-e8m0` branch,
`primus_turbo/hip/grouped_gemm_mxfp8/` — shipped hybrid autograd (HIP fwd +
HIP dgrad + Triton wgrad) 2026-04-23.

## Result summary (MI355X, gpt_oss_20B MoE gate_up: M=65536, K=2880, N=5760, G=32 balanced)

| path          | Triton TFLOPS | HIP TFLOPS | ratio |
|---------------|---------------|------------|-------|
| fwd kernel    | 1423          | **1610**   | 1.131×|
| dgrad kernel  | 1574          | **2006**   | 1.274×|
| wgrad kernel  | 1291          | (Triton)   | —     |
| step k=1 (ms) | ~11.0 (est)   | **8.997**  | ~1.22×|
| step k=8 with prequant (ms) | ~6.92  | **6.019**   | ~1.15×|

Correctness: 28.44–28.46 dB SNR on all three gradients (= fp8 e4m3 noise
floor). Matches Triton SNR bit-for-bit.

## Foundation: reuse, don't rewrite

The production HIP single-GEMM kernel
(`csrc/kernels/gemm/turbo/turbo_gemm_mxfp8_kernel.h` — the
`GEMM_Tile_MXFP8_NT_256x256x128_16x16x128_4_WAVE_GFX950` struct) is already
compute-side optimal for this problem:

- 512 MFMA insts with only 27 s_waitcnt total (0.053 waitcnt/MFMA vs Triton's
  ~0.875) → 16× fewer serialization fences
- C accumulator pinned in AGPR via `reserve_agpr_range<0, 255>`
- `buffer_load_lds` direct-to-LDS (0 `ds_write` in the dump)
- Double-buffered A/B/scale tiles in LDS

Wrapping it with per-expert flat-grid dispatch (block_to_expert lookup) is a
**~500-line Python+CPP delta** that inherits all the compute-side work.

Skill cross-reference: see `skills/flat_grid_block_to_expert_lookup.json`.

## Key HIP-specific patterns used

### 1. Flat-grid flat-tile dispatch (reuse of fwd kernel body)
```cpp
const int32_t expert_g = block_to_expert[blockIdx.x];
const int32_t tile_base = tile_offs[expert_g];
const int32_t local_tile = blockIdx.x - tile_base;
const int32_t pid_m_l = local_tile / tiles_n;
const int32_t pid_n_l = local_tile % tiles_n;
// Then exact copy of single-GEMM kernel body with:
//   m_g    = group_offs[g+1] - group_offs[g]
//   a_base = a + (group_offs[g] + pid_m) * K
//   b_base = b + (g*N + pid_n) * K   // B in NT [G, N, K]
```

### 2. Preshuffle pad for K % 128 ≠ 0
The stock `preshuffle_scale_16x4_kernel` covers only `(scale_cols/4)*4`
columns. For K=2880 → scale_cols=90 (not ÷4), the last 2 scale cols stay
uninitialised and produce garbage on the tail K-iter.

**Fix**: a padded variant that zero-fills cols up to `round_up(scale_cols, 4)`.
Since e8m0 0x00 = 2^-127 ≈ 0, the tail scales null out the tail-K contribution
which BufferSRD already zeroed for data.

```cpp
template <typename InT, typename OutT>
__global__ void preshuffle_scale_16x4_padded_kernel(
    const InT *in, OutT *out, int rows, int cols_real, int cols_padded) {
    // ...
    if (col < cols_real && (bid*16 + row) < rows) val = in[row * cols_real + col];
    else                                         val = 0;
    out[i * 16 * 4 + tid] = val;
}
```

Pass `scale_cols_padded = (scale_cols + 3) & ~3` as the stride to the GEMM kernel.

### 3. Dgrad via fwd-kernel reuse (no new kernel needed)
`dA = dC @ B` with `B_orig [G, N, K]` NT:
- Pre-transpose B to `[G, K, N]` once (`quant_mxfp8_weight_dgrad` already does
  this as its default layout)
- Call fwd kernel with kernel-roles:
  - `A_k = dC`          (kernel sees M=M_total, K_k=N_fwd)
  - `B_k = b_dgrad`     (kernel sees N_k=K_fwd, K_k=N_fwd)
  - Output `[M, K_fwd]` = dA ✓

**1.274× over Triton dgrad** on gpt_oss_20B — bigger win than fwd because
reduction dim (N_fwd=5760) is longer, amortising per-iter overhead.

### 4. Triton scale tensor dtype reinterpret (free)
Triton's `quant_mxfp8_*` returns scales as `torch.uint8`; the HIP binding
`TORCH_CHECK`s for `torch.float8_e8m0fnu`. Same bytes, different tag — zero-
cost reinterpret in Python:
```python
if a_scale.dtype == torch.uint8:
    a_scale = a_scale.view(torch.float8_e8m0fnu)
```

### 5. `cpp_extension.load` flags for Primus-Turbo headers
PyTorch cpp_extension sets `-D__HIP_NO_HALF_OPERATORS__=1` /
`-D__HIP_NO_HALF_CONVERSIONS__=1` by default, which breaks Primus-Turbo's
`float8.h` conversions. Override:
```python
extra_cuda_cflags=[
    "-O3", "--offload-arch=gfx950", "-std=c++17",
    "-U__HIP_NO_HALF_OPERATORS__",
    "-U__HIP_NO_HALF_CONVERSIONS__",
]
```

## PMC findings (diagnostic, for future tuning)

On primary shape (gpt_oss_20B):

| metric                    | HIP grouped | Triton grouped |
|---------------------------|-------------|----------------|
| VGPR                      | 256 (occ=1) | 124            |
| AGPR (from ASM dump)      | Yes, used   | No (Triton)    |
| LDS static                | 144 KB      | (dynamic)      |
| SQ_INSTS_LDS              | 27.1M       | 52.0M          |
| SQ_WAIT_INST_LDS          | **2.9M**    | **48.5M**      |
| LDS stall ratio           | 10.7%       | **93.3%**      |
| MFMA busy (MfmaUtil)      | 44.5%       | 33.5%          |
| VALU util                 | 82.7%       | 98.6%          |
| GRBM_GUI_ACTIVE (cycles)  | 19.6M       | 26.4M          |

**Triton's 93% LDS stall ratio is the scale-LDS waitcnt pathology**. HIP's
10.7% is the compute-floor (waitcnts around barriers between pipeline
phases).

**HIP is NOT compute-ceiling**: MfmaUtil=44.5% means MFMA engine idle ~55% of
the wall. Why? LDS=144KB pins occupancy=1 (gfx950 LDS/CU = 160KB). With
occ=1 there's no latency hiding across waves. To get >44.5% MfmaUtil, need
either:
- Shrink LDS (smaller tile / single-buffer — hurts pipeline)
- Switch to 32×32×64 MFMA (halves inst count; same compute, fewer gaps)
- Persistent kernel amortising launch/edge costs across experts

## Hybrid autograd

Current shipped path:
- fwd:   HIP (1.131× Triton)
- dgrad: HIP via dgrad-layout B (1.274× Triton)
- wgrad: **Triton** variable-K (HIP wgrad deferred — see next section)

Prequant: `MXFP8WeightPrequantHip` hoists per-forward B quant to once per
optim-step. 1.495× step speedup on gpt_oss_20B at k=1 (8.997ms → 6.019ms).
Matches Triton's 1.22× → 1.53× lever within 3%.

## HIP wgrad — v0 shipped, v1 deferred

### v1.3 + v1.4 + v1.5 (current state, 2026-04-24)

**v1.3 — fast fp8 permute** (`_permute_fp8.py::fp8_permute_M_to_GN`):
PyTorch's `.permute(...).contiguous()` for fp8 runs at 12% HBM peak. LDS-tile
Triton transpose kernel hits 85%. 9× speedup on 377 MB go permute (1534→168 us).
This moved HIP wgrad from 0.44× → 0.99× Triton kernel-only. **Key lesson**:
always profile before assuming the kernel is the bottleneck — 60% of HIP
wgrad time was in a torch API call.

**v1.4 — HIP wgrad as autograd default** (per user directive):
Removed `torch.equal(group_offs, expected)` CPU-GPU sync in the HIP entry
point (was forcing per-call host sync). Hybrid+prequant step: 6.24 ms =
1.17× pure Triton (6.24 vs 7.28 ms).

**v1.5 — padded unbalanced fallback**
(`grouped_gemm_mxfp8_hip_variable_k_padded`): per-expert HIP calls with
pad-to-128 (min 384), skips zero-token experts. Correctness: 28.46 dB on
all 8 real gpt_oss_20B shapes. Perf: 4-5× slower than Triton on normal
unbalanced (32 per-expert launches dominate), **0.67× Triton on catastrophic
warmup** (only 4 non-zero experts — padded HIP wins). Available as opt-in;
not default.

### Fused-quant attempt (failed, documented)

Tried `quant_mxfp8_dual_jagged_permuted` — fuse the fp8 permute into the
Triton quant kernel. Bit-exact correctness but 20% SLOWER (1301 vs 1092 us)
due to Triton's BlockedEncoding fighting the permuted-store pattern. See
skill `triton_permuted_store_blocking_layout_pitfall`.

### Final numbers (gpt_oss_20B, MI355X)

| kernel         | Triton ms | HIP ms | ratio    |
|----------------|----------:|-------:|---------:|
| fwd (256 tile) | 1.56      | 1.38   | **1.14×** |
| dgrad (reuse)  | 1.42      | 1.10   | **1.29×** |
| wgrad (v1.3)   | 1.68      | 1.69   | 0.99×    |

| step              | ms   | vs Triton |
|-------------------|-----:|----------:|
| pure Triton       | 7.28 | 1.00×     |
| hybrid (no preq)  | 9.20 | 0.79×     |
| **hybrid+prequant** | **6.24** | **1.17×** |

Correctness: 28.46 dB on out/grad_a/grad_b (= fp8 e4m3 floor).

### v0 → v1 (early work, shipped)

**v0 (per-expert loop)**: `grouped_gemm_mxfp8_hip_wgrad` reused the fwd kernel
per-expert — G=32 launches. **7.733 ms = 0.215× Triton.** Launch overhead
dominated (~50us × 32 = 1.6ms).

**v1 (single-launch, current)**: key insight — for balanced MoE, M_g is uniform
so `group_offs_stacked = [0, N, 2N, ..., G*N]` + A_kern `[G*N, M_g]` + B_kern
`[G, K, M_g]` handles all experts in ONE fwd-kernel call.
**4.100 ms = 0.404× Triton (1.88× over v0).**

Optimization attempts that DID NOT help:
- `quant_mxfp8_dual` (row+col outputs): 5.04 ms — wasted row writes.
- `quant_mxfp8_colwise_for_variable_k` + fp8 permute: 4.84 ms — colwise quant slower than rowwise on this shape.

Balanced MoE only, M_g ≥ 384, M_g % 128 == 0.
Correctness: 28.46 dB (fp8 floor); bit-consistent with Triton.

### v2 (purpose-built kernel, deferred)
Remaining 2.48× gap to Triton comes from the **fwd kernel at wgrad shape**:
tile 256×256×128 tuned for fwd (M=65536, N=5760, K=2880) drops to ~690 TFLOPS
at wgrad shape (M_stacked=G*N=184320, N=K=2880, K_red=M_g=2048) — 43% of fwd
efficiency. 2880 is tile-unaligned (2880%256=64), short K_red loses
compute/tile amortization.

v2 = dedicated variable-K kernel with jagged scales + LDS transpose per
`primus_turbo/hip/grouped_gemm_mxfp8/WGRAD_DESIGN.md`. Expected outcome:
**parity with Triton**, not speedup — Triton wgrad is memory-bound (stall
ratio ~0.91), and waitcnt-reduction (the HIP compute-side lever) does not
move the HBM ceiling.

Engineering cost: ~2-3 days for LDS-transpose tile load + jagged-scale
preshuffle + variable-K prologue/epilogue + <384-K fallback.

### Decision rule
**Only port to HIP if Triton's kernel is compute-bound.** Profile first with
PMC (`SQ_WAIT_INST_LDS` / `SQ_INSTS_LDS` ratio). See skill
`pmc_before_kernel_rewrite`.

## Files / references

- HIP kernel + launcher + pybind: `primus_turbo/hip/grouped_gemm_mxfp8/turbo_grouped_gemm_mxfp8.hip`
- Python wrapper: `primus_turbo/hip/grouped_gemm_mxfp8/__init__.py`
- Hybrid autograd + prequant: `primus_turbo/hip/grouped_gemm_mxfp8/autograd.py`
- Env selector: `TURBO_MXFP8_GG_BACKEND=hip|triton` (default hip on gfx950)
- Tests: `test_phase_a.py` (fwd), `test_phase_b.py` (dgrad), `test_autograd.py` (full)
- Wgrad future-work design: `WGRAD_DESIGN.md`
- Reference single-GEMM (reused tile struct): `csrc/kernels/gemm/turbo/turbo_gemm_mxfp8_kernel.h`
