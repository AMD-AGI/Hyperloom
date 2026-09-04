---
title: Radeon 8060S / gfx1151 — topology, WGP/CU layout, UMA placement
kind: hardware
topic: topology
gens: [gfx1151]
updated: 2026-09-02
---

# Topology — WGPs, CUs, SIMD32s and one UMA node

This card combines AMD's RDNA3.5 execution hierarchy with the live KFD topology of the
qualified EVO-X2. It replaces the XCD/chiplet assumptions used by Instinct cards.

## Live KFD identity

`/sys/class/kfd/kfd/topology/nodes/1/properties` reports:

| KFD property | Live value | Interpretation |
|---|---:|---|
| `gfx_target_version` | `110501` | native `gfx1151` |
| `simd_count` | 80 | total SIMD32 engines exposed |
| `simd_per_cu` | 2 | two SIMD32s per logical CU |
| derived CU count | **40** | `80 / 2` |
| `array_count` | 4 | four live arrays reported by KFD |
| `simd_arrays_per_engine` | 2 | live array grouping |
| `cu_per_simd_array` | 10 | ten CUs per reported SIMD array |
| `wave_front_size` | **32** | native compiled wave size reported to KFD |
| `max_waves_per_simd` | **16** | live wave-slot ceiling |
| `lds_size_in_kb` | **64** | LDS associated with one CU half |
| `max_slots_scratch_cu` | 32 | live scratch-slot property |
| `num_xcc` | **1** | no multi-XCD Instinct topology |
| `mem_banks_count` | 1 | one KFD memory-bank record |
| `drm_render_minor` | 128 | `/dev/dri/renderD128` |
| `vendor_id` / `device_id` | 4098 / 5510 | hex `1002:1586` |

The 40 KFD CUs map architecturally to 20 two-CU WGPs when all pairs are enabled. Use the
KFD **40-CU count** for grid sizing and reporting; use the WGP model for wave/LDS placement.

## Architecture hierarchy

```text
Radeon 8060S / one KFD GPU node / one XCC
└── 20 two-CU WGPs implied by 40 live CUs
    ├── CU0 half: 2 × SIMD32 + 64 KiB LDS half
    └── CU1 half: 2 × SIMD32 + 64 KiB LDS half
        WGP total: 4 × SIMD32 + 128 KiB LDS / 64 banks
```

AMD's RDNA3.5 model says:

- one WGP has four SIMD32s;
- the WGP is logically split into two CU halves;
- all waves of one work-group stay on the same WGP;
- wave placement and LDS visibility depend on CU mode versus WGP mode;
- a work-group may contain at most 1024 work-items;
- the WGP can hold multiple work-groups, subject to waves, registers and LDS.

## CU mode versus WGP mode

| Mode | Wave placement | LDS view | Important restriction |
|---|---|---|---|
| **CU mode** | all waves of a work-group remain on one CU's two SIMD32s | one 64 KiB LDS half | upper/lower halves cannot share; direct/parameter LDS loads are available |
| **WGP mode** | waves may use all four SIMD32s across both CUs | contiguous 128 KiB WGP LDS | `LDS_PARAM_LOAD` and `LDS_DIRECT_LOAD` are unavailable |

One work-group may still allocate at most **64 KiB**, even though WGP mode exposes a 128 KiB
address space. Do not infer a 128 KiB per-work-group budget.

## UMA placement

This is an integrated GPU:

```text
CPU cores ─┐
GPU/WGPs ──┼── shared memory controllers/channels ── LPDDR system memory
other DMA ─┘
```

Consequences:

- GPU global memory and CPU memory pressure draw from one physical capacity.
- GTT accounting and process RSS can overlap; never add them as independent memory pools.
- CPU traffic, page residency, display use and co-tenants can reduce available GPU bandwidth.
- There is no HBM stack, XCD-local HBM, NPS mode or CPX slice to target.
- The PCI function identity is not evidence that model data traverses a discrete PCIe VRAM link.

## Grid sizing

Start from **40 live CUs**, then account for the kernel:

```text
resident workgroups
  <= wave-slot capacity
  <= VGPR/SGPR capacity
  <= LDS capacity
  <= workgroup/dispatch limits
```

Rules:

- Enough workgroups are needed to cover 40 CUs, but no fixed multiple is universally optimal.
- A small decode kernel may be launch/dispatch-bound before it is occupancy-bound.
- A large prefill kernel may prefer fewer, deeper tiled workgroups.
- `nwarps` is a count of waves: on native wave32, `nwarps=4` means 128 threads.
- Do not reuse an MI350 256-CU or MI300 304-CU grid heuristic.

## No chiplet-locality claims

The live node reports `num_xcc=1`. Therefore do not apply:

- per-XCD L2 placement or XCD CTA swizzles;
- XCD-count tile multiples;
- CPX/DPX/SPX or NPS partition recommendations;
- cross-XCD clock-spread explanations;
- HBM stack locality.

Locality still matters at WGP/LDS/cache/page/channel levels, but it must be measured with
Strix-specific counters or controlled data-layout experiments.

## What it means for kernels

1. Size launch breadth from 40 CUs, not from a donor GPU name.
2. Choose CU/WGP behavior deliberately when LDS communication spans waves.
3. Keep a work-group's LDS allocation ≤64 KiB.
4. Treat wave32 thread geometry as the native baseline.
5. Keep CPU/co-tenant traffic controlled during memory-sensitive A/B tests.
6. Do not claim chiplet locality unless a future device exposes and proves such topology.

## Verify

```bash
cat /sys/class/kfd/kfd/topology/nodes/*/properties
cat /sys/class/drm/card1/device/vendor
cat /sys/class/drm/card1/device/device
readlink -f /sys/class/drm/renderD128/device
```

For runtime APIs, compare KFD against `hipGetDeviceProperties` values such as
`multiProcessorCount`, `warpSize`, shared memory and total visible memory. A discrepancy must
be recorded rather than silently choosing the convenient value.

## Sources

- AMD RDNA3.5 ISA guide, Chapters 1–3 and 12.
- Live KFD/sysfs topology on the qualified EVO-X2.

## Related

`gfx1151_overview.md` · `gfx1151_execution.md` · `gfx1151_lds.md` ·
`gfx1151_memory.md` · `gfx1151_clocks.md`
