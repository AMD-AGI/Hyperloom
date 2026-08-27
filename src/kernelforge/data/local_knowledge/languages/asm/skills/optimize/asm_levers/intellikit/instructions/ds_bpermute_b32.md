---
instruction: ds_bpermute_b32
category: memory
architecture: gfx950
tags: [ds_bpermute_b32, lgkmcnt, cross-lane, softmax, permutation]
---

# ds_bpermute_b32

Cross-lane data permutation via LDS hardware: reads lane `addr/4` from the source VGPR and writes to the current lane's destination VGPR. Does not actually use LDS memory.

## Quick Facts

| Property | Value |
|----------|-------|
| Encoding | DS |
| Latency CPI | 2.52 ref-clk (median 1290 ticks / 512 iters) |
| Normalized latency | ~14.7 shader cycles |
| Category | Cross-lane / LDS |

## Hazards

Counts against the lgkmcnt counter. Requires `s_waitcnt lgkmcnt(0)` before consuming the result.

## Known Bugs / Gotchas

### The efficient half-wave exchange for softmax

The forward attention kernel uses exactly 2 `ds_bpermute_b32` per loop iteration -- one for the max exchange and one for the sum exchange between half-waves (lanes 0-31 and lanes 32-63):

```asm
; Exchange max with partner lane (lane XOR 32):
ds_bpermute_b32 v92, v99, v85   ; v99 = (lane ^ 32) * 4 (precomputed)
s_waitcnt lgkmcnt(0)
v_max3_f32 v89, v121, v85, v92  ; combine half-wave maxes
```

This replaces the 80-bpermute butterfly reduction used by naive implementations. Combined with `v_max3_f32` lane-local reduction, the total softmax max reduction is 18 instructions (16 v_max3_f32 + 1 bpermute + 1 v_max3_f32) vs 192 instructions in V4.

### Address operand is byte offset / 4

The `addr` operand specifies the source lane as `lane_id * 4` (byte address into a virtual 256-byte space, 4 bytes per lane). Precompute the partner lane address:

```asm
; Precompute partner address for lane XOR 32:
v_xor_b32 v99, 32, v_lane_id   ; partner = lane ^ 32
v_lshlrev_b32 v99, 2, v99      ; byte address = partner * 4
```

### High variance in measured latency

The isa-bench measurement shows 35.1% CV (coefficient of variation), with median 1290 ticks but max 2996 ticks. This suggests `ds_bpermute_b32` latency varies significantly depending on the permutation pattern and LDS bank conflicts.

### Not a true LDS operation

Despite using the DS encoding and counting against lgkmcnt, `ds_bpermute_b32` does not access LDS memory. It uses the LDS hardware's crossbar for lane-to-lane data movement. This means it does not conflict with actual LDS reads/writes and does not consume LDS bandwidth.

## Common Usage Patterns

### Half-wave max exchange (attention softmax)
```asm
; After lane-local max via v_max3_f32 chain:
ds_bpermute_b32 v_partner_max, v_partner_addr, v_lane_max
s_waitcnt lgkmcnt(0)
v_max3_f32 v_global_max, v_running_max, v_lane_max, v_partner_max
```

### Half-wave sum exchange (attention softmax)
```asm
; After lane-local sum via v_add_f32 chain:
ds_bpermute_b32 v_partner_sum, v_partner_addr, v_lane_sum
s_waitcnt lgkmcnt(0)
v_add_f32 v_global_sum, v_lane_sum, v_partner_sum
```

### Cross-lane address computation (GGEMM direct-to-LDS)
```asm
; 8 ds_bpermute_b32 per iteration in persistent FWD pingpong variant
; Used to shuffle load addresses across lanes for coalesced access
```

## Sources

- Empirically validated on MI355X
- 2 bpermutes per softmax iteration (max + sum exchange) is a common pattern
- Pingpong variants use 8 bpermutes for address shuffle
- isa-bench: crosslane_bpermute latency kernel
