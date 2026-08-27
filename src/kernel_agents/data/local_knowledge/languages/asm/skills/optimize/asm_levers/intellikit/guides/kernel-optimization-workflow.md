---
guide: kernel-optimization-workflow
category: methodology
architecture: gfx950
tags: [workflow, optimization, disassembly, round-trip, profiling, Triton, assembly, methodology, roofline]
---

# Kernel Optimization Workflow for gfx950

How to write, optimize, and validate hand-tuned assembly kernels on MI355X. Covers the three main approaches, the iteration loop, and hard-won rules from production kernel development.

---

## Table of Contents

1. [Three Approaches](#1-three-approaches)
2. [Recommended: Start From Reference](#2-recommended-start-from-reference)
3. [The Round-Trip Workflow](#3-the-round-trip-workflow)
4. [Correctness Validation](#4-correctness-validation)
5. [Performance Measurement](#5-performance-measurement)
6. [The Optimization Loop](#6-the-optimization-loop)
7. [Common Pitfalls](#7-common-pitfalls)
8. [When to Use Each Approach](#8-when-to-use-each-approach)

---

## 1. Three Approaches

### A: Write Assembly From Scratch

Start with an empty `.s` file. Write the kernel descriptor, metadata, and every instruction by hand.

**Pros:**
- Full control over every detail
- No inherited inefficiencies
- Deep understanding of the kernel

**Cons:**
- Extremely slow (days to weeks for a complex kernel)
- Must solve every correctness challenge from zero: MFMA layouts, LDS addressing, softmax precision, register allocation, NOP hazards
- High risk of subtle bugs that take days to debug

**When to use:** Simple kernels (GEMV, elementwise, reductions) or when no reference exists.

### B: Start From Triton

Write the kernel in Triton, compile it, disassemble the `.co`, then hand-optimize the assembly.

**Pros:**
- Triton handles high-level correctness (tiling, reductions, softmax)
- Generates a working `.co` to start from
- Easy to iterate on the Triton source for algorithmic changes

**Cons:**
- Triton's codegen may use suboptimal patterns (extra barriers, VGPR-staged LDS loads instead of direct-to-LDS)
- Disassembly round-trip introduces issues (DPP instructions dropped, branch offsets instead of labels)
- Must understand Triton's hidden args and metadata to launch reassembled kernels

**When to use:** New kernel types where you want Triton to handle the high-level algorithm, then you hand-optimize the hot loop.

### C: Start From Reference Assembly (Recommended)

Take a known-fast implementation (from an optimized library, a Triton compilation, or a prior iteration), disassemble it, verify the round-trip, then make targeted modifications.

**Pros:**
- Starts from proven-correct, proven-fast code
- Inherits all the hard-won optimizations (scheduling, register allocation, LDS layout)
- Incremental changes are easy to validate — diff against the baseline
- Fastest path to a working optimized kernel

**Cons:**
- Must understand the reference code deeply before modifying
- Inherited complexity can be opaque
- Risk of cargo-culting patterns you don't understand

**When to use:** Always, when a reference exists. This is the approach that has produced the best results.

---

## 2. Recommended: Start From Reference

The most successful workflow across 25+ kernel optimization campaigns:

```
1. Get the reference .co (compiled binary)
2. Disassemble it to .s
3. Clean up the .s (labels, formatting)
4. Reassemble to .co
5. Verify the round-trip produces identical results
6. THEN modify and optimize
```

**Why this wins:** Writing from scratch burned days debugging basic correctness (MFMA output layout, LDS transpose addressing, softmax numerical precision) that the reference already handles correctly. The reference is the ground truth — learn from it, don't reinvent it.

**Rule:** Never optimize until you have a bit-identical round-trip. If your unmodified reassembly doesn't match the original, you don't have a valid baseline.

---

## 3. The Round-Trip Workflow

### Step 1: Disassemble

```bash
/opt/rocm/llvm/bin/llvm-objdump -d --mcpu=gfx950 reference.co > raw_disasm.s
```

The raw disassembly has numeric branch offsets, no labels, and may drop some instructions (especially DPP modifiers). Clean it up:

```bash
python3 disasm_to_asm.py raw_disasm.s > kernel.s
```

`disasm_to_asm.py` converts numeric branch targets to labels and reformats for reassembly.

### Step 2: Reassemble

```bash
/opt/rocm/llvm/bin/llvm-mc --triple=amdgcn-amd-amdhsa --mcpu=gfx950 \
  -filetype=obj kernel.s -o kernel.o
/opt/rocm/llvm/bin/ld.lld -shared kernel.o -o kernel.co
```

On some ROCm versions, paths may be `/opt/rocm-7.2.0/lib/llvm/bin/` instead of `/opt/rocm/llvm/bin/`.

### Step 3: Patch

If your `.s` doesn't include the full kernel descriptor and metadata (common when starting from disassembly), splice just the `.text` section into the reference ELF:

```bash
python3 patch_co.py --ref reference.co --new kernel.co -o patched.co
```

This preserves the original kernel descriptor, metadata, and `.args` section while replacing only the code.

**Size constraint:** If the new `.text` is larger than the reference, the patch corrupts the ELF. Verify sizes match or your code is smaller (NOP padding is harmless).

### Step 4: Validate

```python
import torch
ref_output = run_kernel("reference.co", inputs)
new_output = run_kernel("patched.co", inputs)
cos = torch.nn.functional.cosine_similarity(
    ref_output.flatten().float(),
    new_output.flatten().float(),
    dim=0
)
print(f"cos_sim = {cos.item():.6f}")  # Must be 1.000000 for round-trip
```

**Round-trip must produce cos_sim = 1.000000.** Not 0.999999 — exactly 1.0. Any deviation means the disassembly→reassembly lost or changed instructions.

### Step 5: Baseline Performance

```python
# Warmup
for _ in range(20):
    run_kernel("patched.co", inputs)
torch.cuda.synchronize()

# Measure
import time
start = time.perf_counter()
for _ in range(100):
    run_kernel("patched.co", inputs)
torch.cuda.synchronize()
elapsed = (time.perf_counter() - start) / 100
```

The patched kernel should have identical performance to the reference. If it's more than 1% different, the round-trip is not clean.

---

## 4. Correctness Validation

### Thresholds

| Metric | Round-Trip | Optimized vs Reference |
|--------|-----------|----------------------|
| cos_sim | == 1.000000 | >= 0.999990 |
| max_abs_error | == 0.0 | < 1e-2 (BF16), < 1e-4 (FP32) |
| allclose | True | True (atol depends on dtype) |

### Validation Rules

1. **Always compare against the reference output, never against a mathematical formula.** The reference may have intentional precision tradeoffs (e.g., exp2 instead of exp, fast rsqrt).

2. **Test multiple inputs.** Random data, edge cases (zeros, large values, negative), and real model weights. Some bugs only manifest with specific data patterns.

3. **Test multiple shapes.** The kernel may be correct at M=128 and wrong at M=256 due to tiling edge cases.

4. **Test multiple dtypes.** BF16, FP16, FP32. Format conversion bugs often hide in single-dtype testing.

5. **Re-assemble before comparing.** Never trust a pre-existing `.co` on a remote node across work sessions. Always re-assemble from the current `.s` source to avoid chasing phantom divergences from stale binaries.

---

## 5. Performance Measurement

### Methodology

1. **Warmup:** 20-50 iterations (fills caches, stabilizes clocks)
2. **Measured:** 100+ iterations with torch.cuda.synchronize() at start and end
3. **Report:** Median of 5 runs, not mean (avoids outlier skew)

### Roofline Analysis

Every performance result should be contextualized against the hardware roofline:

```
MI355X Roofline:
  HBM Bandwidth: ~5.3 TB/s
  Peak FP16 MFMA: ~2517 TFLOPS (measured effective)
  LDS Bandwidth: ~high (64-bank, conflict-free)
```

For a GEMM at shape (M, N, K):
```
FLOPs = 2 * M * N * K
Bytes = (M*K + K*N) * sizeof(dtype) + M*N * sizeof(output_dtype)
Arithmetic Intensity = FLOPs / Bytes
```

If AI > machine_balance → compute bound (compare against peak TFLOPS)
If AI < machine_balance → memory bound (compare against peak BW)

### Profiling

When a kernel underperforms, profile before guessing:

```bash
rocprofv3 --hip-activity --hsa-activity -o profile.csv ./run_kernel
```

Look at:
- **Occupancy** (waves/SIMD): Check `next_free_vgpr` against occupancy breakpoints
- **Wait fraction**: High lgkmcnt waits → LDS bottleneck; high vmcnt waits → memory bottleneck
- **MFMA utilization**: Low utilization → scheduling or NOP overhead

---

## 6. The Optimization Loop

Once you have a validated round-trip baseline:

```
1. Identify the bottleneck (profile, roofline)
2. Read how the incumbent handles it
3. Form a hypothesis (scheduling change, instruction upgrade, register reallocation)
4. Make ONE change
5. Reassemble and validate correctness
6. Measure performance
7. If better: keep, push, document in changelog
8. If worse or neutral: revert, document why
9. Repeat
```

### High-Value Optimizations (Rough Priority)

| Optimization | Typical Gain | Effort |
|-------------|-------------|--------|
| MFMA opcode upgrade (16x16x16 → 16x16x32) | 20-80% | Low |
| Direct-to-LDS (`buffer_load ... lds`) | 10-17% | Medium |
| Register pressure reduction (cross occupancy threshold) | 3-33% | High |
| Software pipelining (double/triple buffer) | 10-30% | High |
| NOP scheduling (co-execute useful work during NOPs) | 2-5% | Medium |
| `s_setprio 3` around MFMA blocks | 0.5-1% | Low |
| Barrier elimination (wave-0-only spin) | 3-6% | Medium |

### The "Read the Damn Code" Rule

When stuck, clone the reference implementation and read the actual source:
- What MFMA opcode does it use?
- How does it layout LDS?
- How does it handle the B-matrix transpose?
- What's its register budget?
- How does it schedule memory loads relative to compute?

The answers are in the code, not in documentation.

---

## 7. Common Pitfalls

### Pitfall 1: Optimizing Before Validating
**Symptom:** "28% faster" kernel that produces zeros.
**Cause:** Dropped 608 instructions during disassembly round-trip (184 DPP shuffles + MFMAs).
**Rule:** ALWAYS validate correctness before measuring performance.

### Pitfall 2: Stale .co Baseline
**Symptom:** All variants show cos_sim = 0.97 against baseline.
**Cause:** The baseline `.co` was assembled from an older `.s` version.
**Rule:** Re-assemble baseline from current source before every comparison.

### Pitfall 3: Claiming Victory Too Early
**Symptom:** Kernel passes at one shape, fails at another.
**Cause:** Tiling edge cases, register lifetime bugs that only manifest at larger K.
**Rule:** Test at least 3 shapes, 2 dtypes, with both random and structured inputs.

### Pitfall 4: Over-NOPping
**Symptom:** Kernel is correct but 15% slower than reference.
**Cause:** Using 4x `s_nop 15` everywhere when 2x suffices for your context.
**Rule:** Start conservative (4x), validate, then reduce NOPs and re-validate.

### Pitfall 5: Ignoring Disassembly Artifacts
**Symptom:** Reassembled kernel is 184 instructions shorter.
**Cause:** DPP `quad_perm` modifiers silently dropped by `llvm-objdump`.
**Rule:** Count instructions in original vs reassembled `.co`. They must match.

---

## 8. When to Use Each Approach

| Kernel Type | Recommended Approach | Reasoning |
|------------|---------------------|-----------|
| GEMM (BF16/FP16/FP8) | Reference (from optimized library) | Mature references exist with deep scheduling |
| Attention (FWD/BWD) | Reference (from optimized library) | Softmax precision, MFMA layout, LDS swizzle are hard to get right |
| GEMV / Reductions | From scratch or Triton | Simple enough that assembly from scratch is viable |
| Elementwise (SiLU, RMSNorm) | From scratch | Straightforward memory-bound kernels |
| Grouped/Variable-K GEMM | Triton → optimize | Triton handles the variable-K dispatch, optimize the inner loop |
| Uber-kernel (fused ops) | Reference + compose | Take optimized building blocks, fuse with inter-WG barriers |
| Novel architecture | Triton → optimize | Let Triton prototype the algorithm, then hand-tune |

### Decision Flowchart

```
Does a fast reference implementation exist?
├── Yes → Start from reference assembly
│         (disassemble → round-trip → optimize)
└── No  → Is the algorithm complex (attention, multi-stage GEMM)?
          ├── Yes → Write in Triton first, compile, then optimize assembly
          └── No  → Write assembly from scratch
                    (GEMV, elementwise, reductions)
```

---

## Appendix: Lab Notebook Discipline

Push early, push often. Maintain a timestamped CHANGELOG.md:

```markdown
## 2026-05-09 14:30 — Round-trip validated
- Disassembled reference.co (1847 instructions)
- Reassembled: 1847 instructions, .text size matches
- cos_sim = 1.000000, timing within 0.5%

## 2026-05-09 16:00 — MFMA opcode upgrade
- Replaced 16x16x16_bf16 with 16x16x32_bf16
- Required: new LDS addressing (16x32 thread-to-K mapping differs)
- Result: 1.4x speedup, cos_sim = 0.999998
- Pushed as V2

## 2026-05-09 18:30 — Direct-to-LDS (REVERTED)
- Converted buffer_load+ds_write to buffer_load_lds
- Saved 16 VGPRs, eliminated 4 ds_write + 1 barrier
- Result: 12% speedup BUT cos_sim = 0.985 at M=512
- Root cause: m0 not updated between loads
- Reverted, will fix m0 management tomorrow
```

Every modification gets a changelog entry. The journey matters as much as the result.
