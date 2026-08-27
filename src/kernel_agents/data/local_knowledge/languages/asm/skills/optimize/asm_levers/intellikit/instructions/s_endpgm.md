---
instruction: s_endpgm
category: scalar
architecture: gfx950
tags: [s_endpgm, vmcnt, store-leak, s_waitcnt]
---

# s_endpgm

Terminate the shader program. Deallocates all resources (VGPRs, SGPRs, LDS) and signals the command processor that the wave has finished.

## Quick Facts

| Property | Value |
|----------|-------|
| Encoding | SOPP |
| Category | Scalar flow control |

## Hazards

### Does NOT drain outstanding stores

On gfx950, `s_endpgm` does NOT guarantee that in-flight `buffer_store` or `global_store` operations have committed to memory. Outstanding stores may be silently dropped ("store leak").

## Known Bugs / Gotchas

### MANDATORY: s_waitcnt vmcnt(0) before s_endpgm

Every gfx950 kernel MUST have `s_waitcnt vmcnt(0)` immediately before `s_endpgm`. Without it, store operations issued in the epilogue may not complete, causing partial or missing output data.

```asm
; WRONG -- stores may leak:
buffer_store_dwordx2 v[10:11], v1, s[4:7], 0 offen
s_endpgm

; CORRECT:
buffer_store_dwordx2 v[10:11], v1, s[4:7], 0 offen
s_waitcnt vmcnt(0) lgkmcnt(0)   ; drain ALL outstanding memory ops
s_endpgm
```

This was identified as a mandatory correctness fix across attention and GEMM kernels. The reference Triton kernels sometimes omit this, creating a latent correctness hazard.

### Symptoms of missing waitcnt

- Intermittent correctness failures (sometimes correct, sometimes partially corrupted)
- More visible under load or when running back-to-back kernel launches
- Output buffer contains stale data from previous kernel launch
- Non-deterministic -- may pass on one run and fail on the next

### Trailing NOP sled

Triton-compiled kernels often have 50-300 `s_nop 0` instructions after `s_endpgm` as ELF alignment padding. These never execute. When patching assembly to add instructions elsewhere, these trailing NOPs can be removed to maintain constant `.text` section size (required by the ELF patcher).

## Common Usage Patterns

### Standard kernel exit
```asm
; After all stores:
s_waitcnt vmcnt(0) lgkmcnt(0)
s_endpgm
; NOP sled for alignment (never executes):
s_nop 0
s_nop 0
; ...
```

### Persistent kernel exit
```asm
; After outer loop exhausts all tiles:
.L_exit:
  s_waitcnt vmcnt(0) lgkmcnt(0)
  s_endpgm
```

## Sources

Independently confirmed across GEMM and attention kernel families on MI355X.
