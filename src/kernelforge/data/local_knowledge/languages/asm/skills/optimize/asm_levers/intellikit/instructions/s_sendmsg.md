---
instruction: s_sendmsg
category: scalar
architecture: gfx950
tags: [s_sendmsg, GS_DONE, dealloc_vgprs, wave-completion, message]
---

# s_sendmsg

## Syntax

```asm
s_sendmsg <message_id>
```

## Description

Sends a message from a shader wave to the hardware fixed-function units. On gfx950, the primary use case is `GS_DONE` and `DEALLOC_VGPRS` signaling.

## Variants

| Message | Value | Use |
|---------|-------|-----|
| `sendmsg(MSG_DEALLOC_VGPRS)` | 0x10 | Release VGPRs early before wave completion |
| `sendmsg(MSG_GS_DONE)` | 0x3 | Signal geometry shader completion |

## VGPR Deallocation

`s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)` releases the wave's VGPRs back to the hardware allocator before `s_endpgm`. This allows the next wave to begin allocating VGPRs sooner, reducing inter-wave latency.

```asm
s_waitcnt vmcnt(0) lgkmcnt(0)
s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)
s_endpgm
```

**Constraint:** All memory operations must be drained (s_waitcnt) before sending DEALLOC_VGPRS. Any in-flight load that writes to a deallocated VGPR will fault or produce silent corruption.

## Counter

No counter. This is a fire-and-forget message to the hardware message bus.

## Known Issues

1. **Must follow s_waitcnt.** VGPRs become invalid after the message — any pending load/store writing VGPRs will corrupt.
2. **Not always beneficial.** On compute-bound kernels at max occupancy, the wave teardown is already fast enough that DEALLOC_VGPRS provides no measurable benefit. Most useful for short kernels where launch overhead dominates.
