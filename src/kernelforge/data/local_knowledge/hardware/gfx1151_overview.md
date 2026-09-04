---
title: Radeon 8060S / Strix Halo — gfx1151 orientation and cheat sheet
kind: hardware
topic: overview
gens: [gfx1151]
updated: 2026-09-02
---

# Radeon 8060S / Strix Halo (gfx1151) — orientation

**Start here for the EVO-X2 target.** This card separates three evidence classes:

- **ISA facts** from AMD's RDNA 3.5 ISA guide;
- **live platform facts** from KFD/sysfs on the qualified EVO-X2;
- **measured local observations** from retained gfx1151 campaigns.

Do not import MI300/MI350 values merely because both stacks use ROCm.

## One-screen cheat sheet

| Fact | Radeon 8060S / EVO-X2 value | Why it matters |
|---|---|---|
| ISA target | **`gfx1151`**, compile with `--offload-arch=gfx1151` | Never spoof gfx1100/gfx1101/gfx942/gfx950 |
| Architecture | **RDNA 3.5**, integrated Radeon 8060S | WMMA/VOPD model, not CDNA MFMA |
| PCI identity | AMD `1002:1586` on this node | Confirms the physical device selected by KFD |
| Native wave | **wave32**; wave64 is supported | The same `nwarps` means half the threads of a wave64 design |
| Live compute topology | **40 CU**, **80 SIMD32**, 2 SIMD/CU | Size grids from 40 CUs, not MI tables |
| Live wave-slot cap | **16 waves/SIMD** from KFD | Hard platform ceiling before VGPR/LDS limits |
| WGP model | 2 CU / 4 SIMD32 per WGP | Work-groups and 128 KiB WGP LDS live at this level |
| LDS | **64 KiB/CU**, **128 KiB/WGP**, 64 banks; one work-group ≤64 KiB | Re-derive every shared-memory tile/swizzle |
| Matrix family | 16×16×16 **WMMA**: F16, BF16, IU8, IU4 | `AMD_WMMA_AVAILABLE`; CDNA `MFMA` is not the route |
| Wave32-only lever | **VOPD** dual issue | Real opportunity, but strict bank/source/destination rules |
| Memory model | **UMA/shared LPDDR**, not dedicated HBM | CPU, weights, KV, GTT and GPU share channels/capacity |
| Host RAM on qualified node | 132,565,487,616 B physical (~123.5 GiB) | Dynamic availability matters more than nominal capacity |
| Local bandwidth reference | 256 GB/s theoretical; ~241 GB/s read and ~209 GB/s copy measured | Node-specific roofline anchor, not an RDNA3.5 guarantee |
| Qualified software baseline | Torch 2.13.0+rocm10.0.0, HIP 7.15.26333, native gfx1151 | Bind claims to the exact container/toolchain |

## What is architectural and what is not

### Architectural

- RDNA3.5 supports wave32 and wave64.
- A WGP contains four SIMD32s and two CU halves.
- LDS is 128 KiB/WGP over 64 banks; one work-group may request at most 64 KiB.
- VOPD is legal only for wave32.
- WMMA includes F16/BF16 floating forms and IU8/IU4 integer forms.
- FLAT participates in both global and LDS dependency domains.

### Qualified EVO-X2 node

- KFD reports `gfx_target_version=110501`, `simd_count=80`, `simd_per_cu=2`,
  `wave_front_size=32`, `max_waves_per_simd=16`, `lds_size_in_kb=64`, and `num_xcc=1`.
- The device is PCI `1002:1586` and exposes render node 128.
- The system uses shared DDR/LPDDR capacity rather than discrete VRAM/HBM.

### Measured, not universal

- The 256 GB/s theoretical and ~241/~209 GB/s stream results belong to this memory
  configuration and probe methodology.
- Model PP/TG results, selected launch geometries and route wins belong to their exact model,
  quantization, batch, context and runtime.
- A software route working on gfx1151 does not prove the format is a native matrix dtype.

## The deltas that break Instinct/CDNA ports

1. **wave64 → native wave32** — lane collectives, masks, threads/block and occupancy change.
2. **MFMA → WMMA/VOP3P** — operand layouts and instruction families are different.
3. **HBM → UMA** — CPU and GPU contend for the same memory channels and capacity.
4. **CDNA LDS rules → RDNA WGP/CU modes** — 128 KiB/WGP, two 64 KiB halves, 64 banks.
5. **XCD topology → one live gfx1151 node/XCC** — no per-XCD L2 or CPX/NPS assumptions.
6. **Instinct libraries are conditional** — AITER/CK kernels must prove exact gfx1151 support.
7. **FP8/MX claims do not transfer** — validate the selected framework loader and physical route.

## Roofline discipline

Do not publish a compute ridge from an unverified peak-FLOP number. For this target:

- establish achievable memory bandwidth with an on-box probe;
- measure actual device time and bytes moved;
- inspect whether the target path is WMMA, dot, VALU, Triton or a framework fallback;
- report prefill and decode separately;
- include host/dispatch time for small and decode-dominated workloads.

The retained large-model decode evidence often approaches the shared-memory bandwidth ceiling,
but that does not classify every prefill, attention, fusion or quantization shape.

## Portable rules for gfx1151

- Compile natively for `gfx1151`; never use `HSA_OVERRIDE_GFX_VERSION` as support evidence.
- Assume wave32 unless the exact code object proves deliberate wave64.
- Derive workgroup geometry from 40 live CUs and the kernel's resource metadata.
- Use RDNA WMMA/VOPD rules; do not recommend CDNA MFMA instructions.
- Keep one work-group at or below 64 KiB LDS and reason about all 64 banks.
- Treat GTT and process RSS as views of shared physical memory, not additive pools.
- Require route, correctness and code-object evidence before interpreting throughput.
- Keep framework/version/image qualifications narrow.

## Verify on box

```bash
rocminfo | grep -m1 -oE 'gfx[0-9a-f]+'
cat /sys/class/kfd/kfd/topology/nodes/*/properties
cat /sys/class/drm/card1/device/{vendor,device}
hipcc --version
```

For generated kernels, retain:

- native gfx1151 code-object identity;
- wave size and launch geometry;
- VGPR/SGPR/LDS/private/spill metadata;
- target instruction/route proof;
- correctness results and matched runtime measurements.

## Sources

- AMD, *RDNA 3.5 Instruction Set Architecture Reference Guide*:
  https://docs.amd.com/v/u/en-US/rdna35_instruction_set_architecture
- AMD/GPUOpen machine-readable ISA:
  https://gpuopen.com/machine-readable-isa/
- Live KFD/sysfs topology on the qualified EVO-X2.
- Local measured reference: `evo-gfx1151-mmvq-tuning/references/gfx1151-architecture-and-benchmarks.md`.

## Related

`gfx1151_topology.md` · `gfx1151_execution.md` · `gfx1151_matrix_core.md` ·
`gfx1151_dtypes.md` · `gfx1151_lds.md` · `gfx1151_memory.md` ·
`gfx1151_isa.md` · `gfx1151_clocks.md`
