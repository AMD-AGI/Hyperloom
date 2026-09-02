---
title: Radeon 8060S / gfx1151 — UMA memory hierarchy and bandwidth discipline
kind: hardware
topic: memory
gens: [gfx1151]
updated: 2026-09-02
---

# Memory hierarchy — UMA, caches and shared bandwidth

The Radeon 8060S is an integrated GPU. Its global memory ultimately resides in shared system
LPDDR/DDR rather than dedicated HBM. CPU traffic, GPU weights/KV, display activity and other DMA users
can contend for the same physical channels and capacity.

## Qualified-node memory facts

| Fact | Qualified EVO-X2 value | Evidence class |
|---|---:|---|
| Physical system memory | 132,565,487,616 B (~123.5 GiB) | live Linux memory total |
| GPU memory model | UMA/shared system memory | platform architecture/live behavior |
| Theoretical memory interface reference | 256 GB/s | node configuration calculation |
| STREAM-style HIP read | ~241 GB/s | measured local probe |
| STREAM-style copy | ~209 GB/s | measured local probe |
| Dedicated HBM | none | platform fact |
| KFD memory-bank records | 1 | live KFD topology |

The measured numbers are a ceiling reference for that probe, size, clock/thermal state and software
stack. They are not guaranteed application bandwidth.

## Architecture ladder

AMD's RDNA3.5 model contains:

```text
VGPRs
  ↕
LDS (128 KiB/WGP, 64 banks; one work-group ≤64 KiB)
  ↕
per-WGP/on-chip caches and device cache hierarchy
  ↕
L2 channels / memory controllers
  ↕
shared system memory on the integrated platform
```

The ISA describes read-only L1/L0 paths, write-combining/atomic-cache behavior and cache-control bits,
but cache capacities and platform MALL details are device-specific. Do not copy MI350 Infinity Cache,
per-XCD L2 or HBM capacities into gfx1151 documentation without direct device evidence.

## UMA accounting

- GTT is an addressable GPU view backed by shared system pages, not a second physical RAM pool.
- Process RSS, page cache and GPU allocations can overlap in physical accounting.
- `MemAvailable` is more useful for admission than nominal total RAM.
- Swap pressure can destabilize latency and consume memory bandwidth.
- CPU memory-intensive work can lower GPU throughput even without owning `/dev/kfd`.
- Model weights, KV, activations and runtime/JIT buffers compete with the host OS and co-tenants.

For a large-model admission decision, record at least:

```text
MemTotal / MemAvailable / swap use
process RSS and cgroup limits
KFD/GTT ownership
model weights + KV + runtime estimate
memory-pressure and I/O-PSI state
```

Never report `RSS + GTT` as independent physical consumption without reconciling overlap.

## Bandwidth and roofline

For a memory-bound kernel:

```text
time_floor ≈ bytes transferred / achievable bandwidth
```

Use the measured ~241 GB/s read ceiling only as a node-specific comparison point. Compute implied
bandwidth from actual tensor/model bytes, then label omitted traffic such as:

- KV and activation reads/writes;
- quantization scales/metadata;
- page tables and cache misses;
- output stores;
- allocator/copy traffic.

Do not publish a FLOP/byte ridge until the exact instruction path's sustainable compute rate is
measured or authoritatively established.

## Decode versus prefill

- **Decode/GEMV/small-batch:** often streams weights per token and can approach the shared-memory wall;
  fixed dispatch/host overhead is also significant.
- **Prefill/large-M GEMM:** offers more reuse and may become compute, LDS or scheduling limited.
- **Attention:** moves KV and can change classification with context length, page layout and cache dtype.
- **MoE:** expert routing reduces active weight traffic but introduces irregularity and launch overhead.

Never combine PP and TG into one “GPU bandwidth” claim.

## GLOBAL, BUFFER and FLAT

- Use contiguous, aligned GLOBAL/BUFFER vector accesses when the address space is known.
- FLAT participates in both global and LDS dependency domains and may complete through different units.
- A compiler emitting FLAT for a pure tensor-global hot loop is a signal to inspect pointer provenance.
- Buffer descriptors can provide bounds semantics without branch-heavy per-lane guards.
- Cache controls must match reuse/streaming and atomic semantics; measure rather than cargo-culting flags.

## Layout and access

- Put the contiguous tensor dimension across active lanes.
- Preserve vector alignment after padding/slicing.
- Separate global coalescing from LDS bank-conflict analysis.
- Reorder/preshuffle off the hot path only when its storage and conversion cost amortize.
- For one-use data, avoid LDS staging unless it removes another larger cost.
- Include tails and non-power-of-two dimensions in correctness and performance tests.

## What it means for kernels

1. Count bytes and launches before optimizing arithmetic.
2. Compare implied bandwidth against an on-box achievable probe, not HBM tables.
3. Control CPU/co-tenant activity for memory-sensitive A/B tests.
4. Include packing, scale and copy traffic in low-bit claims.
5. Use global-address instructions when pointer provenance permits.
6. Treat large shared-memory capacity as admission headroom, not guaranteed bandwidth.
7. Monitor swap/PSI during long campaigns.

## Pitfalls

- Calling shared GTT “VRAM” and adding it to RAM.
- Importing MI350 HBM/L2/Infinity Cache numbers.
- Treating theoretical 256 GB/s as application-sustained bandwidth.
- Inferring exact bytes/token from model file size alone.
- Benchmarking while CPU memory traffic or another iGPU job is active.
- Calling a small-model gap purely memory efficiency without separating launch overhead.
- Assuming a source vector type produced a wide memory instruction.

## Verify

- Re-run a size-swept read/copy microbenchmark in the same runtime and thermal state.
- Inspect device ISA for access width, address space and cache controls.
- Capture device time, host wall time and dispatch count separately.
- Record `free`, swap and PSI during the run.
- Confirm model/request route, context, KV dtype and batch geometry.

## Sources

- AMD RDNA3.5 ISA guide, memory/cache, VMEM, GLOBAL/FLAT/SCRATCH and LDS chapters.
- Live Linux/KFD data from the qualified EVO-X2.
- Retained STREAM-style gfx1151 roofline measurements.

## Related

`gfx1151_topology.md` · `gfx1151_lds.md` · `gfx1151_execution.md` ·
`gfx1151_clocks.md` · `common_methodology/optimization/lever_coalescing.md`
