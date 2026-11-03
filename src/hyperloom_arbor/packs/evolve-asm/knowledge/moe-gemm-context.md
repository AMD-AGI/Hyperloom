# DSv4.1-Flash MoE GEMM Context

## Problem
DeepSeek-V4.1-Flash on MI355X is 21% behind B200 at C32, 16% at C1.
Root cause: no tuned FlyDSL tile config exists for these shapes in aiter.
The fallback heuristic tiles are leaving significant performance on the table.

## Shapes (after TP=8 sharding)

| Kernel | Operation | M_total | N | K | dtype_act | dtype_w |
|--------|-----------|---------|---|---|-----------|---------|
| moe1 decode | gate+up fused | batch×6 | 4608 | 5120 | FP8 e4m3fn | FP4 e2m1 |
| moe2 decode | down proj | batch×6 | 5120 | 2304 | FP8 e4m3fn | FP4 e2m1 |
| moe1 prefill | gate+up fused | up to 16384 | 4608 | 5120 | FP8 e4m3fn | FP4 e2m1 |
| dspark moe | draft MoE | batch×3 | 4608 | 5120 | FP8 e4m3fn | FP4 e2m1 |

Experts=384, top_k=6 (main MoE); experts=128, top_k=3 (DSpark).

## Current fallback tiles (from vLLM startup logs)

- Decode: `flydsl_moe1_afp8_wfp4_bf16_t32x128x256_w2_gui`
- Prefill: `flydsl_moe1_afp8_wfp4_bf16_t64x128x256_w4_bnt0_gui`

## FlyDSL tile config knobs (TIER 1 — exhaust these first)

```
tile_m:    32, 64, 128          # CTA M dimension — try 64 first for decode
tile_n:    128, 256             # CTA N dimension
tile_k:    128, 256             # K reduction tile
k_wave:    1, 2, 4, 7          # waves per CTA — k_wave=7 often wins for MoE
gui_mode:  True/False           # global unswizzled indexing
bnt:       0, 1, 2              # bank-conflict avoidance mode
persist:   True/False           # persistent kernel
async_copy: True/False          # async global→LDS
```

**Start with: tile_m=64, k_wave=7, gui_mode=True** — historically the highest-leverage combo for MoE on gfx950.

## Roofline context (moe1 M=32, FP4 weights)

- Arithmetic intensity: 2×32×4608×5120 / (4608×5120×0.5 + 32×5120 + 32×4608×2 + ...) ≈ 12 FLOP/B
- Ridge point (gfx950 MAF FP4): 5e15 / 6.4e12 ≈ 781 FLOP/B
- At M=32 the kernel is **heavily memory-bound** (AI << ridge)
- At M=32 we are also **occupancy-limited** — too few workgroups for 256 CUs
- Lever: smaller tile_m (more CTAs → better CU coverage) OR split-K

## Integration

On KEEP: run bench driver with `--deploy --config-name <name> --tile-config <tile>`
This writes to the aiter CSV which is git-tracked on the host.
The model will automatically pick up the tuned config on next server start.
