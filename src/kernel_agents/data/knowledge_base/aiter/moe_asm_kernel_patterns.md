# Fused MoE ASM Kernel Patterns (gfx942/gfx950)

Hand-written AMDGPU ISA kernel for `fmoe_bf16_blockscale_fp8` — fused
MoE MLP block with dynamic FP8 re-quantization between the gate+up
GEMMs and the down GEMM. Extracted from AMD-internal design doc
`fused_moe_kernel_in_asm_for_amdgpu.md` (mi300/mi350 target). Cross-
refs: [`asm_kernel_patterns.md`](asm_kernel_patterns.md),
[`../hip/mfma_and_register_ops.md`](../hip/mfma_and_register_ops.md).

## When to apply

- Expert FFN block in an MoE model: `Y = down( act(X·G^T) * (X·U^T) )`
  with 128×128 block-scaled FP8 weights and per-token FP8 X.
- Single thread group per expert tile, memory-bound on weight HBM
  bandwidth more than on MFMA throughput.
- You need HBM traffic for `X` halved across the gate+up pair.

## Kernel geometry

| Knob             | Value                                                         |
|------------------|---------------------------------------------------------------|
| Thread group     | 1 TG, 256 threads, 4 wavefronts (`CNT_WV = 4`)                |
| Output tile      | `sub_X × sub_GU` = 32 tokens × 128..512 inter_dim             |
| Sub-tile today   | 32×256 is the only shipped configuration                      |
| `SUB_X`          | 32 (tokens processed per TG)                                  |
| `CNT_X_Q`        | `SUB_X / 16 = 2` (token blocks per thread)                    |
| Block scale      | 128×128 FP8, with optional U1 upscale                         |
| K split          | halves `K=0..3` and `K=4..7` (eight K-fragments)              |
| Prefetch depth   | `PF_Bs = 2` (double-buffered weight loads)                    |
| MFMA type        | FP8 × FP8 → FP32 <!-- confirm: exact tile shape not stated --> |

All three GEMM stages run inside one persistent TG — there is no
launch between gate+up and down.

## Stage pipeline

```
X_FP8  ──┐
         ├─► GEMM0 (Z = X·G^T)  ─► dequant(sX*sG) ─┐
G_FP8  ──┘                                         │
                                                   ├─► act() ─► Z′_FP32
X_LDS  ──┐                                         │
         ├─► GEMM1 (GM1 = X·U^T)─► dequant(sX*sU) ─┘
U_FP8  ──┘                                         │
                                                   ▼
                          (LDS-MAX reduce, FP8 re-quant of Z′·GM1)
                                                   │
                                                   ▼
                                 GEMM2 (down)      <!-- confirm: down-proj detail not in doc -->
```

Only GEMM0+GEMM1 fusion and the intermediate re-quant are described
in depth in the source doc. The down-projection (GEMM2) is named in
the summary but its register plan is not specified.

## GEMM0 / GEMM1 fusion

Both GEMMs share the same `X` operand. The kernel loads `X` **once**
into LDS during GEMM0 (the memory-feeder half) and **reuses it from
LDS** during GEMM1. This halves X HBM traffic versus two independent
GEMMs.

```
GEMM0_withup(cl_p, pi, pad):
  for token_block in sub_X (32):         # outer, over token stripes
    for j in CNT_GU_ADDR (e.g. 8):       # hidden_dim block
      for i in CNT_X_Q (2):              # token block
        # K split in two halves, alternating G+U issue:
        K = 0..3:  load_U; load_X; MFMA Z += G·X
        K = 4..7:  load_U; load_X; MFMA Z += G·X
    dequant Z  with (XQ × GQ) via DPP broadcast   → Z_fp32
  # barrier; X is now resident in LDS
  for token_block in sub_X:
    for j, i:
      K = 0..3:  load_G; load_GQ; load_X_LDS; load_XQ_LDS; MFMA GM1 += U·X
      K = 4..7:  load_G;           load_X_LDS; load_XQ_LDS; MFMA GM1 += U·X
    dequant GM1 with (XQ × UQ) via DPP broadcast → GM1_fp32
```

Key point: GEMM0 issues `load_U` and `load_X` interleaved (two
separate HBM streams for latency hiding), whereas GEMM1 only issues
`load_G` from HBM — `X` comes from LDS with near-zero cost.

## K split in two halves (dual MFMA stream)

Inner K (depth 8, in units of MFMA K-fragment) is partitioned into
`K=0..3` and `K=4..7`. Each half-pass runs its own MFMA fragments
with its own load→compute cadence. This:

- Gives the memory unit two independent load fronts that can overlap
  with the other half's MFMA.
- Lets the compiler/asm author shuffle wait counts per-half without
  stalling the whole K loop.
- Interleaves G-column groups so XQ/UQ scale loads can be tucked at
  the end of the first half and consumed at the start of the second.

## Dequant via DPP broadcast

After the accumulator is full for a `(j,i)` block, per-block
quantization scales (`XQ`, `GQ` / `UQ`) are applied **in-register**.
Scales are 128-element blocks; the kernel uses DPP
(`__builtin_amdgcn_mov_dpp` / cross-lane permute) to broadcast one
lane's scale into the 16 or 32 lanes that own the same block, then
does `fp32 *= scale` on the MFMA output tile.

- `Z  = Z  × (XQ × GQ)` after GEMM0.
- `GM1 = GM1 × (XQ × UQ)` after GEMM1.
- `XQ × GQ` and `XQ × UQ` are combined scalars per 128×128 block, so
  one fmul per element after the broadcast.

The DPP route means *no LDS round-trip for scales* — scales are
loaded once into 1..4 VGPRs per thread and permuted.

## Activation fusion

Between GEMM0+GEMM1 and the re-quant/GEMM2 stage the kernel fuses
`act(Z) * GM1` (SwiGLU / GeGLU shape):

```
Z  = act(Z)            # SiLU or GELU, in fp32
Z′ = Z * GM1           # elementwise gated mul
```

`Z′` then enters the LDS-MAX reduction below. Because both Z and GM1
live in pinned fp32 registers (`_v_Z`, `_v_GM1`), the activation is
pure VALU — no round-trip to LDS or HBM.

## Dynamic FP8 re-quant of the GEMM0+1 output

Down-projection expects FP8. The kernel dynamically picks a
per-(token-block, hidden-stripe) scale via an LDS reduction:

1. Each thread computes local `abs-max` across its 128-element slice
   using `v_max3_f32` (3-operand max fused pair). Init with `1e-6` to
   dodge div-zero.
2. 16 threads per wave hold two MAX values each (`CNT_X_Q = 2`);
   write to LDS with `ds_write_b64` at
   `LDS_GMAX_BASE + wave_id*512 + (tid & 0xf)*8`.
3. `s_waitcnt lgkmcnt(0); s_barrier`.
4. Every participating thread reads back all 16 threads' MAXes with
   `ds_read_b64` in a 16-iteration unrolled loop (offsets `8*16*i`).
5. Local reduction with `v_max3_f32` down to `_v_Max[0], _v_Max[1]`.
6. `scale = DENO / max` (DENO = 240.0 for FP8, 127.0 for INT8) using
   `v_rcp_f32` + `v_mul_f32`.
7. `Z *= scale` then `v_cvt_pk_fp8_f32` packs two fp32 into one fp8
   pair.
8. Store the inverse `ZdynQ = max / DENO` for the dequant side on
   the GEMM2 output path.

LDS footprint for this reduction:
`LDS_GMAX_SIZE = CNT_WV × SUB_X × 4 × 4 = 512 dwords per wave`
× 4 waves = 2048 dwords ≈ 8 KB. Only 16 of 64 lanes per wave
participate (`tid & 0xf`).

## Register aliasing

The kernel intentionally aliases register ranges across phases to
stay under the VGPR budget. Named buffers in the doc:

```
_v_X      X matrix tile (from HBM or LDS)
_v_G      G / U weights (alternating owner across phases)
_v_Z      GEMM0 fp32 accumulator
_v_GM1    GEMM1 fp32 accumulator
_v_XQ     X per-block scales
_v_GQ     G per-block scales
_v_UQ     U per-block scales
_v_Max    per-thread abs-max (2 entries)
_v_GMAX   gathered per-wave MAX values (32 entries in 16 threads)
_v_ZdynQ  stored dequant scale for later stage
```

`_v_G` is re-pointed between GEMM0 (gate) and GEMM1 (up) so the same
physical VGPR window holds whichever weight is live. After the
dequant of Z, the MFMA A-operand registers that held `G` fragments
are reused as the `_v_Max` scratch and later as `_v_GMAX` gather
buffer. Exact VGPR/AGPR counts are not stated in the doc.
<!-- confirm: specific VGPR count and aliasing map -->

## LDS plan

| Region        | Purpose                                        | Lifetime               |
|---------------|------------------------------------------------|------------------------|
| `X` tile      | loaded once in GEMM0, read in GEMM1           | until GEMM1 dequant    |
| `LDS_GMAX`    | 2048 dwords, per-wave abs-max scratch          | GEMM0+1 re-quant only  |
| Weight DBs    | `PF_Bs = 2` prefetch buffers for G / U        | rolling across K loop  |

X stays resident across both gate and up passes; this is the
primary LDS-reuse win. Weight double-buffers use `PF_Bs = 2`
(explicit in the doc) to overlap the next K-fragment's HBM load with
the current fragment's MFMA.

## Wait / barrier pattern

Only barriers explicitly called out in the doc:

- Prologue of each GEMM0+1 outer pass: "Wait for X in LDS; wait for
  G blocks in memory" — a vmcnt/lgkmcnt wait, then `s_barrier`.
- MAX reduction: `s_waitcnt lgkmcnt(0)` before the `s_barrier`, then
  another barrier before the LDS read fan-out. This is the only
  hard sync between GEMM1 done and GEMM2 start.
- Within a K-split half, `s_waitcnt vmcnt(N)` is used to keep a
  bounded number of outstanding weight loads (exact `N` not stated).
<!-- confirm: vmcnt threshold -->

## Performance tuning knobs (per doc)

| Knob                | Value / range                | Notes |
|---------------------|------------------------------|-------|
| `SUB_X`             | 32                           | Only value shipped |
| `sub_GU`            | 128..512                     | 256 is the shipped path |
| `sub_D` (down K)    | matches `sub_GU`             | <!-- confirm: not explicit --> |
| K inner depth       | 8, split as 4+4              | Hard-coded in ISA path |
| `CNT_GU_ADDR`       | e.g. 8                       | Hidden-dim blocks per token stripe |
| `CNT_X_Q`           | `SUB_X/16 = 2`               | Token blocks per thread |
| `CNT_WV`            | 4                            | Waves per TG |
| `PF_Bs`             | 2                            | Weight prefetch depth |
| Block scale         | 128×128                      | FP8 with optional U1 upscale |
| LDS MAX region      | 2048 dwords (~8 KB)          | Only re-quant scratch |
| FP8 DENO            | 240.0 (127.0 for INT8)       | Saturation target |
| Abs-max init        | `1e-6`                       | Prevents div-zero downstream |

## Pitfalls

- **Do not reissue X from HBM in GEMM1.** The whole X-traffic halving
  depends on keeping X in LDS across the gate→up boundary. A naive
  port that reloads X re-doubles HBM traffic.
- **DPP broadcast of scales is layout-sensitive.** Scales ride in one
  lane per 128-wide group; if the MFMA output layout changes (e.g.
  switching between 16×16 and 32×32 tiles) the lane that owns the
  canonical scale moves. Rederive the broadcast lane before changing
  the MFMA shape. See the related layout-vs-broadcast gotcha logged
  in user memory under `feedback_flydsl_mfma32_layout.md` (gfx950
  MFMA 32x32x16 bf16 per-lane output layout). <!-- confirm -->
- **Abs-max must be initialized to a small nonzero (`1e-6`).** A zero
  input block would drive `scale = 240/0` and NaN the whole stripe.
- **K-split halves are not independent GEMMs** — both halves write
  into the *same* `_v_Z` / `_v_GM1` accumulator. If you move MFMAs
  across the half boundary you must keep the dependence chain.
- **`_v_G` is aliased between gate and up.** Any prologue that reads
  `_v_G` assuming it still holds `G` after GEMM0 is a correctness
  bug. The dequant of Z must be complete before GEMM1 issues its
  first `U` load into that window.
- **LDS MAX scratch is per-wave, not per-lane.** Only `tid & 0xf`
  writes; the other 48 lanes per wave are idle for the reduction.
  A refactor that spreads work across all 64 lanes must also respect
  the 16×wave layout on read.
- **Only 32×256 is shipped.** Other `sub_GU` values (128, 384, 512)
  are named in the matrix-dim section but not validated.

## Cross-reference

- General AITER ASM kernel conventions:
  [`asm_kernel_patterns.md`](asm_kernel_patterns.md).
- MFMA tile / lane layout tables + AGPR pinning rule:
  [`../hip/mfma_and_register_ops.md`](../hip/mfma_and_register_ops.md).
- Related skills: `moe_gemm0_gemm1_fusion`,
  `moe_blockscale_dpp_dequant`, `moe_k_split_dual_stream`,
  `moe_register_aliasing`, `moe_lds_x_reuse`.
