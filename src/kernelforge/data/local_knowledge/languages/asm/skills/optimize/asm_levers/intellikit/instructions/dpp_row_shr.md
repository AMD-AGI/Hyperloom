---
instruction: dpp_row_shr
category: vector
architecture: gfx950
tags: [dpp_row_shr, DPP, cross-lane, broken, no-op]
---

# DPP row_shr (Data-Parallel Primitives row shift right)

DPP modifier that shifts data right within a 16-lane row. Applied as a modifier to VALU instructions (e.g., `v_add_f32 dst, src0, src1 row_shr:N`).

## Quick Facts

| Property | Value |
|----------|-------|
| Encoding | DPP modifier on VOP2/VOP1 |
| Category | Cross-lane / DPP |

## Hazards

N/A -- this modifier is non-functional on gfx950.

## Known Bugs / Gotchas

### COMPLETELY BROKEN on gfx950

DPP `row_shr` modifiers on `v_add_f32` (and likely other VALU instructions) are a **complete no-op at runtime** on gfx950. The instruction assembles correctly, encodes correctly, and executes without error -- but the DPP modifier has no effect. The result is as if the DPP modifier was not present (i.e., the instruction operates on same-lane data only).

```asm
; DOES NOTHING -- row_shr modifier is silently ignored:
v_add_f32 v4, v4, v4 row_shr:1 bound_ctrl:0
; v4 ends up as v4 + v4 (same lane), NOT v4 + v4_from_lane_below
```

This was discovered during softmax cross-lane reduction implementation. The butterfly reduction pattern using DPP row_shr produced no cross-lane movement at all. The kernel produced correct results only because the reduction was accumulating the same value (same lane) repeatedly, which happened to give a plausible (but wrong) output.

### Workaround: v_readlane_b32 loop

The standard workaround for cross-lane reductions on gfx950 is a `v_readlane_b32` loop:

```asm
; Reduction: sum 64 lanes of v4
s_nop 1              ; hazard guard if VALU just wrote v4
v_readlane_b32 s_tmp, v4, 0
v_mov_b32 v_acc, s_tmp
v_readlane_b32 s_tmp, v4, 1
v_add_f32 v_acc, v_acc, s_tmp   ; or use SGPR accumulation
; ... repeat for all 64 lanes
```

### Alternative: ds_bpermute_b32 for specific patterns

For half-wave exchange (lane XOR 32) as used in softmax, `ds_bpermute_b32` is more efficient than 64 readlane iterations:

```asm
; Exchange with partner lane (lane XOR 32):
ds_bpermute_b32 v_partner, v_partner_addr, v_value
s_waitcnt lgkmcnt(0)
v_max3_f32 v_result, v_prev, v_value, v_partner
```

### Scope of breakage

It is not confirmed whether ALL DPP modifiers are broken or only `row_shr`. Other DPP operations (row_shl, row_ror, row_bcast, quad_perm) have not been extensively tested on gfx950 on this architecture. Assume all DPP modifiers are suspect until proven otherwise on gfx950.

## Common Usage Patterns

### DO NOT USE on gfx950

Use `v_readlane_b32` for cross-lane data extraction to SGPR, or `ds_bpermute_b32` for lane-to-lane permutation.

## Sources

- Empirically validated on MI355X: DPP row_shr confirmed non-functional at runtime
- v_readlane_b32 loop validated as replacement
- gfx950 reference: DPP anti-pattern
