---
title: compile checks — cross-ISA builds, VGPR / LDS / spill metadata and grid residency
kind: technique
gens: [gfx942, gfx950]
updated: 2026-11-09
sources:
  - sgl-project/sgl-amd-mono@219f40be:sgl_amd_mono/minimax_m31/residency.py
  - sgl-project/sgl-amd-mono@219f40be:sgl_amd_mono/minimax_m31/runner.py
  - sgl-project/sgl-amd-mono@219f40be:sgl_amd_mono/common/plan.py
  - https://github.com/sgl-project/sgl-amd-mono/pull/19
---

# Compile checks

A persistent launch has two hard limits a timing run will not explain: every CTA must be resident at
once, and nothing may spill. Both are read off the compiled code object, and a code object for an ISA
this host does not have can still be built.

## Build for another ISA (FlyDSL 0.4.0)
```bash
COMPILE_ONLY=1 ARCH=gfx950 FLYDSL_GPU_ARCH=gfx950 FLYDSL_RUNTIME_CACHE_DIR=/tmp/xc_cache \
  python3 build_one_launch.py
```
`build_one_launch.py` builds the launch as its runner does (for MiniMax-M3.1:
`ffn.build_ffn(ffn.FfnBuild(tokens=rows, tp=4))`) and calls `flyc.compile(launch, *dummy_args, stream)`.
Nothing runs; each compiled launch lands in the cache as `launch_*/*.pkl`. Use a fresh cache directory
per check so an earlier build cannot answer for this one.

## Read the resources
```python
import glob, pickle, re
for path in sorted(glob.glob("/tmp/xc_cache/launch_*/*.pkl")):
    ir = pickle.loads(open(path, "rb").read()).ir
    name = re.search(r'kernel_metadata<"([^"]+)"', ir).group(1)
    fields = dict(re.findall(r"(\w+) = (\d+) : i\d+", re.search(r"metadata = \{([^}]*)\}", ir).group(1)))
    print(name, {k: v for k, v in fields.items() if any(s in k for s in ("gpr", "spill", "segment"))})
```
`vgpr_count`, `agpr_count`, `sgpr_count`, `vgpr_spill_count`, `sgpr_spill_count`,
`group_segment_fixed_size` (LDS bytes) and `private_segment_fixed_size` (scratch). The `ir` text also
names the target (`#rocdl.target<chip = "gfx950">`), and `residency.py::code_object` extracts the
embedded ELF for a disassembler.

## The limits
- 512 threads a CTA and one CTA a CU is 2 waves a SIMD: at most 256 registers a wave on gfx942 and
  gfx950. A wave at 256 with spills is the signature of a loop that keeps a second copy of its state
  (#19: a guarded wait at 256 VGPRs and 754 spilled, 228 and none once the loop had a single exit).
- LDS per CTA ≤ the CU's LDS (160 KiB gfx950, 64 KiB gfx942). Check every width: LDS pools usually grow
  with rows (MiniMax-M3.1: 66,816 B at 1 row, 163,264 B at 16).
- Residency on a host of the target ISA: `residency.active_workgroups_per_cu(compiled, name, THREADS)`
  asks the HIP driver; `runner.warm` refuses a width when `per_cu × CUs < BLOCKS`. A grid that is not
  resident does not run slowly: its first cross-CTA wait times out.

## Prove a change leaves another ISA alone
Build that ISA's launches before and after the change with the same build keys and compare the
extracted ELF bytes. Do not compare cache keys: `plan.kernel_sources` digests every imported module,
so a key moves when any of them changes, even if the code object does not.
