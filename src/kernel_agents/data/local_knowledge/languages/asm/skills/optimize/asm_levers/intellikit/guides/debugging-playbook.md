---
guide: debugging-playbook
category: debugging
architecture: gfx950
tags: [debugging, correctness, validation, NaN, cos_sim, register-clobber, symptom-diagnosis]
---

# gfx950 Assembly Kernel Debugging Playbook

**Target:** MI355X / CDNA4 / gfx950 assembly kernels (GEMM, attention, grouped GEMM)
**Purpose:** The first document an agent reads when a kernel produces wrong results.
All findings empirically validated on MI355X silicon.

---

## 1. Symptom-to-Cause Lookup Table

When your kernel produces wrong results, match the symptom below to its most likely cause. Causes are ordered by frequency within each symptom.

### Output is All Zeros

| Cause | Mechanism | Diagnostic | Fix |
|-------|-----------|------------|-----|
| AGPR count bug | `next_free_vgpr == accum_offset` in kernel descriptor; hardware allocates 0 AGPRs | Pre-fill output with -1.0; if -1.0 survives, MFMA never wrote | Set `next_free_vgpr = accum_offset + n_agprs` |
| accum_offset aliasing | VGPRs >= accum_offset alias AGPRs; ds_read into v64+ clobbers accumulators | Dump AGPRs via v_accvgpr_read after MFMA; check if zeroed | Map all ds_read destinations below accum_offset |
| Missing store path | Epilogue skipped or s_endpgm before global_store completes | Add `s_waitcnt vmcnt(0)` before `s_endpgm`; check branch logic | Verify store path executes; add waitcnt |
| .set constants produce stub kernel | Using `.set` inside `.amdhsa_kernel` block silently produces no-op kernel | Check .co binary size; disassemble and verify instructions present | Move `.set` directives outside `.amdhsa_kernel` block |

### Output is All NaN

| Cause | Mechanism | Diagnostic | Fix |
|-------|-----------|------------|-----|
| flat/global coherence mismatch | One kernel writes with `global_store`, next reads with `flat_load` (or vice versa); stale L1 data | Check instruction types across kernels; look for sc0/sc1 flags | Use consistent memory types (all global or all flat) throughout pipeline |
| FP16 intermediate overflow | x_buf FP16 intermediate overflows at layer 5+ in rmsnorm/layernorm | Check intermediate values with printf kernel; look for inf propagation | Recompute from FP32 state; avoid FP16 intermediates for reductions |
| LDS reduction bug | Tree reduction in shared memory produces wrong norms, causing division by zero | Bisect: remove LDS reduction, compute per-thread | Fix reduction tree; verify lane participation masks |
| buffer_inv at kernel start | `buffer_inv` invalidates L2 before peer stores land; reads stale/zero data | Remove buffer_inv; check if NaN disappears | Remove buffer_inv; rely on coherence flags (sc0/sc1) instead |
| v_cvt_pk_bf16_f32 omission | Manual BF16 packing via shift produces inf/NaN for denormals and special values | Check epilogue for `v_lshrrev_b32 + v_or_b32` pattern | Use `v_cvt_pk_bf16_f32` exclusively for BF16 packing |

### Output Has ~4x Systematic Error

| Cause | Mechanism | Diagnostic | Fix |
|-------|-----------|------------|-----|
| OCP vs FNUZ FP8 format mismatch | `v_mfma_f32_*_f8f6f4` uses OCP format; data encoded as FNUZ has 2x bias per operand = 4x total | Compare raw hex of FP8 values against OCP encoding table | Apply 0.25x correction factor (0.5x per operand), or re-encode data as OCP |

### Cosine Similarity ~0.0 (Effectively Random Output)

| Cause | Mechanism | Diagnostic | Fix |
|-------|-----------|------------|-----|
| Register mapping fundamentally wrong | MFMA output layout misunderstood (column-major, not row-major) | Use identity matrix input; verify output matches expected layout | `a[k] = C[col_group*4+k, lane_row]` where col_group = VGPR/4, lane_row = lane%16 |
| Assembler produced stub kernel | `.set` inside `.amdhsa_kernel` or metadata error silently produces empty kernel | Disassemble .co; check instruction count; verify KD fields | Fix assembler source; verify with `llvm-objdump -d` |
| Completely wrong kernarg mapping | struct.pack layout doesn't match kernel's sgpr/vgpr kernarg preload | Dump first few SGPRs; compare against expected kernarg values | Match pack format exactly to `.amdhsa_next_free_sgpr` and preload count |

### Cosine Similarity 0.93-0.99 (Close But Not Correct)

| Cause | Mechanism | Diagnostic | Fix |
|-------|-----------|------------|-----|
| Partial register clobber | ds_read or buffer_load destination overlaps live MFMA source/result registers | Map all register lifetimes; identify overlaps | Reassign clobbered registers to dead VGPR range |
| Tail MFMA interleaving error | Moving tail MFMAs before vmcnt drain causes reads of not-yet-loaded data | Test with all tail MFMAs after vmcnt(0) | Respect data dependency: vmcnt must drain before MFMA consumes loaded data |
| ds_read_b128 clobber | Multi-dword LDS read clobbers 4 consecutive VGPRs; may overlap live values | Check v[N:N+3] liveness for every ds_read_b128 destination | Use non-overlapping destination registers |

### Cosine Similarity ~0.71 (Specific Pattern)

| Cause | Mechanism | Diagnostic | Fix |
|-------|-----------|------------|-----|
| Scale factor VGPR clobber | v[138:139] (or similar) holds FP8 scale factor; ds_read_b128 clobbers it mid-loop | Print scale factor value before/after ds_read | Restore scale factor at loop entry; or move to SGPR pair |

### Cosine Similarity ~0.258 (Very Specific)

| Cause | Mechanism | Diagnostic | Fix |
|-------|-----------|------------|-----|
| SGPR buffer descriptor corruption | s6/s7 (or similar descriptor pair) clobbered between tiles by address computation | Dump descriptor SGPRs at each tile boundary | Reload descriptors before each tile; protect descriptor SGPRs from reuse |

### Cosine Similarity ~0.961 (Deterministic, Shape-Independent)

| Cause | Mechanism | Diagnostic | Fix |
|-------|-----------|------------|-----|
| MFMA-ds_write position sensitivity | Moving ds_write relative to MFMA changes timing of LDS data availability | Swap ds_write position; measure cos_sim change | Keep ds_write after its dependent MFMA completes; don't hoist past dependencies |

### Half the Expected Values (cos_sim ~0.5 or output magnitude halved)

| Cause | Mechanism | Diagnostic | Fix |
|-------|-----------|------------|-----|
| Half-K bug | K-loop iterates K/2 times instead of K (off-by-one in loop bound or stride) | Check loop count: should be `K / k_per_mfma_step` | Fix loop bound computation |

### Random Garbage (Non-Deterministic, Changes Each Run)

| Cause | Mechanism | Diagnostic | Fix |
|-------|-----------|------------|-----|
| lgkmcnt FIFO miscount | lgkmcnt drains oldest LDS request first; consuming wrong slot reads incomplete data | Count ds_read instructions between issue and waitcnt; verify FIFO position | Track ds_read FIFO position explicitly; use correct lgkmcnt value |
| vmcnt FIFO ordering error | vmcnt drains oldest global load first; prefetch before data loads inverts expected order | Map all global_load issue points; verify waitcnt values match FIFO drain order | Issue data loads before prefetches; or adjust vmcnt to account for prefetch in flight |
| Missing s_barrier in LDS double-buffer | Waves read LDS while another wave is still writing to same bank | Add s_barrier between LDS write and LDS read phases | Always: `lgkmcnt(0)` then `s_barrier` before cross-wave LDS reads |
| LDS shared region inter-wave race | Prefetch drain writes to shared LDS region before all waves finish reading | Add s_barrier before writes to shared LDS regions | Barrier before write, not just before read |

### Two MFMAs Produce Identical Output

| Cause | Mechanism | Diagnostic | Fix |
|-------|-----------|------------|-----|
| Zero AGPRs allocated | accum_offset == next_free_vgpr; both MFMAs accumulate into same (nonexistent) AGPR space | Check KD: verify accum_offset < next_free_vgpr | Set next_free_vgpr = accum_offset + total_agprs_needed |

### First Run is 2x Faster Than Subsequent Runs

| Cause | Mechanism | Diagnostic | Fix |
|-------|-----------|------------|-----|
| Cold GPU clock boosting | GPU boosts clocks when cold; settles to sustained frequency after thermal ramp | Run 5+ warmup iterations; discard first results | Always warm up; report median of 10+ measured iterations |

### Kernel Launch Fails with Error 701

| Cause | Mechanism | Diagnostic | Fix |
|-------|-----------|------------|-----|
| Missing .args metadata | ROCm 7.2 on gfx950 requires `.args` section in AMDGPU metadata | Check metadata YAML in .s file for `.args:` key | Add `.args:` section with at least kernarg pointer entry |

### Kernel Hangs (Never Returns)

| Cause | Mechanism | Diagnostic | Fix |
|-------|-----------|------------|-----|
| Unbalanced s_barrier | Some waves hit s_barrier, others skip it (e.g., early-exit branch) | Check all control flow paths hit same number of s_barriers | Ensure all waves in workgroup execute identical barrier count |
| Missing s_endpgm | Control flow falls through without hitting s_endpgm | Check all branch targets; verify every path terminates | Add s_endpgm to all exit paths |
| Inter-WG barrier deadlock | Barrier spin-wait with wrong coherence flags; waves never see peer writes | Check for `sc0 sc1` on loads/stores in barrier | Use `flat_atomic sc0` for flag writes, `flat_load sc0 sc1` for flag reads |

### SEGV / Memory Fault

| Cause | Mechanism | Diagnostic | Fix |
|-------|-----------|------------|-----|
| HSA AQL queue wrapping | Queue wrapping bug on ROCm 7.2; direct AQL dispatch SEGVs after queue fills | Limit dispatch count; or use HIP launch path | Use hipModuleLaunchKernel instead of direct AQL dispatch |
| Wrong kernarg base | Kernarg buffer allocated from device memory instead of host memory | Check allocation: must be host-visible (hipHostMalloc or system memory) | Allocate kernarg buffer with hipHostMalloc |
| OOB buffer_load | Buffer descriptor (s[4:7]) has wrong base, stride, or num_records | Dump descriptor SGPRs; verify base address and bounds | Fix descriptor setup; verify num_records covers access range |

---

## 2. Diagnostic Techniques (Ranked by Usefulness)

### Tier 1: Try These First (Minutes to Answer)

**1. Pre-fill Output with -1.0**
Fill the output tensor with -1.0 before launch. If -1.0 survives, the store path or MFMA never executed. If output is zero, MFMA ran but accumulated into wrong location (AGPR bug).
```python
C_gpu.fill_(-1.0)
# launch kernel
result = C_gpu.cpu()
# If any -1.0 remains: that block/tile never wrote output
```

**2. All-Ones Input Test**
Set A = all ones, B = all ones. Expected: C[i,j] = K for all (i,j). If C is all zeros, K/2, or garbage, the bug is structural (not numerical). This eliminates indexing ambiguity since every element should be identical.

**3. Identity Matrix Test**
Set A = I (identity), B = data (or vice versa). Expected: C = B (or A). Tests that your MFMA output mapping and store path preserve the input exactly. Catches column-major vs row-major confusion immediately.

**4. Kernel Descriptor Binary Inspection**
The kernel descriptor is the 64-byte header that controls hardware resource allocation. Mis-encoding silently produces wrong results. Key fields on gfx950:
- **accum_offset**: bits [3:0] of word at offset 0x3C (rsrc3). Value = (field + 1) * 4 VGPRs. If this equals next_free_vgpr, you get zero AGPRs.
- **next_free_vgpr**: controls total VGPR allocation. Must be >= accum_offset + agpr_count.
- **LDS size**: granularity is 128 dwords (512 bytes) on gfx950.
```bash
# Inspect KD from .co binary
python3 -c "
import struct
with open('kernel.co', 'rb') as f:
    data = f.read()
# Find .amdhsa_kernel offset (architecture-dependent)
# Check accum_offset, vgpr count, lds size
"
```

**5. Disassemble and Diff**
When a change breaks correctness, disassemble both .co files and diff:
```bash
llvm-objdump -d --mcpu=gfx950 before.co > before.s
llvm-objdump -d --mcpu=gfx950 after.co > after.s
diff before.s after.s
```
If instruction count differs significantly (e.g., 608 instructions dropped), the assembler silently failed.

### Tier 2: Isolation Tests (Hours to Answer)

**6. Single-MFMA Isolation**
Strip the kernel down to a single MFMA instruction with hardcoded inputs. Verify the output layout matches your mental model. The column-major output layout:
```
a[k] = C[(lane/16)*4+k, lane%16]
```
Where k indexes within the 4-element AGPR group, lane/16 selects the column group, and lane%16 selects the row.

**7. Per-Block Cosine Similarity**
Don't just compute global cos_sim. Compute per-block (per-tile) cos_sim. This localizes the bug to specific workgroups:
```python
for m_block in range(M // BLOCK_M):
    for n_block in range(N // BLOCK_N):
        block_ref = ref[m_block*BM:(m_block+1)*BM, n_block*BN:(n_block+1)*BN]
        block_out = out[m_block*BM:(m_block+1)*BM, n_block*BN:(n_block+1)*BN]
        sim = cos_sim(block_ref.flatten(), block_out.flatten())
        if sim < 0.999:
            print(f"Block ({m_block},{n_block}): cos_sim={sim:.6f}")
```

**8. Test Pyramid (Progressive Complexity)**
Run these in order. Stop at the first failure.
1. **Passthrough**: Load A, store A. Tests kernarg, addressing, store path.
2. **Single tile**: M=BLOCK_M, N=BLOCK_N, K=k_per_step. One MFMA, one tile.
3. **Multi-tile**: M=BLOCK_M, N=BLOCK_N, K=full. Full K-loop, single output tile.
4. **Full shape**: M=256, N=256, K=4096. Multiple workgroups.
5. **End-to-end**: Production shapes (e.g., 138 hipBLASLt shapes for GEMM production).

**9. MFMA Operand Dump**
Before the MFMA instruction, store the A and B operands to a debug buffer. After the MFMA, store the C result. Compare against a Python reference MFMA simulation.

**10. LDS Dump Kernel**
Write a tiny kernel that reads the entire LDS allocation and stores it to global memory. Run after the main kernel to inspect LDS contents. Useful for verifying tile layout, transpose correctness, and bank conflict patterns.

### Tier 3: Deep Investigation (Days to Answer)

**11. Register Lifetime Analysis**
Map every register (VGPR, SGPR, AGPR) from definition to last use. Identify overlaps where a load destination clobbers a live value. This is the most common class of gfx950 assembly bugs.

**12. Python Algorithmic Simulation**
Implement the kernel's algorithm in Python with the same tile sizes, MFMA semantics, and accumulation order. Use this as the ground truth reference. Essential for attention kernels with online softmax.

**13. Per-Wave Breakdown**
For multi-wave workgroups, isolate output per wave. Check if one wave produces correct results while another doesn't. Catches inter-wave synchronization bugs (missing s_barrier, LDS race conditions).

---

## 3. Correctness Thresholds

These thresholds are empirically calibrated from extensive gfx950 kernel development.

### Cosine Similarity Thresholds

| cos_sim | Interpretation | Action |
|---------|---------------|--------|
| >= 0.999999 | **Bit-identical** (within FP rounding) | Correct. Ship it. |
| >= 0.9999 | **Scheduling-level difference** | Acceptable for MFMA reordering. Verify max_diff < 0.03125 (1 ULP for BF16). |
| >= 0.999 | **Minor numerical divergence** | Acceptable for FP8 kernels or different accumulation order. Investigate if unexpected. |
| >= 0.990 | **Persistent kernel baseline** | Non-determinism floor for persistent kernels with atomics. Acceptable only if self-comparison (same kernel, two runs) shows similar variance. |
| >= 0.99 but < 0.999 | **Suspicious** | Likely a partial bug. Check for register clobber, off-by-one in K-loop, or missing waitcnt. |
| 0.95 - 0.99 | **Definitely buggy** | Structural error present. Use per-block cos_sim to localize. |
| 0.885 | **FP8 vs FP32 baseline** | Expected when comparing FP8 MFMA output against FP32 reference. Not a bug if using FP8 throughout. |
| < 0.9 | **Fundamentally broken** | Major structural bug. Start with all-ones test and identity test. |
| ~0.0 | **Random output** | Register mapping wrong, stub kernel, or completely wrong addressing. |

### Absolute Difference Thresholds

| max_diff | Interpretation |
|----------|---------------|
| 0.031250 (1/32) | 1 ULP for BF16. Acceptable for BF16 GEMM. |
| 0.062500 (1/16) | 2 ULP for BF16. Acceptable for FP8 GEMM or attention with online softmax. |
| > 0.125 | Bug. No legitimate numerical difference produces this for normalized inputs. |

### Persistent Kernel Non-Determinism

Persistent kernels (e.g., grouped GEMM FWD with persistent dispatch) have an inherent non-determinism floor:
- **Self-comparison floor**: cos_sim ~0.993-0.997 (same kernel, two consecutive runs, same inputs)
- **Cross-run variance**: cos_sim can drop to ~0.990 across different GPU contexts
- **Implication**: Any cos_sim above the self-comparison floor is "correct" for a persistent kernel

To establish your kernel's non-determinism floor:
```python
# Run kernel twice with identical inputs
result1 = run_kernel(A, B)
result2 = run_kernel(A, B)
floor = cos_sim(result1, result2)
print(f"Non-determinism floor: {floor:.6f}")
# Your optimization is correct if cos_sim(optimized, reference) >= floor
```

---

## 4. "Start From Reference" Methodology

This is the single most reliable way to build a correct gfx950 assembly kernel. Never write from scratch. Always start from a known-working binary.

### Step 1: Extract a Reference Binary

Extract a compiled kernel binary (.co) from an existing library or compiler. Common sources:

```bash
# From any PyTorch-based library that compiles GPU kernels:
# Trigger compilation of target kernel, capture .co from /tmp/

# From rocBLAS (for GEMM reference):
rocblas-bench -f gemm --transposeA N --transposeB T \
  -m 256 -n 256 -k 4096 --a_type bf16 --b_type bf16 \
  --c_type bf16 --d_type bf16 --compute_type f32
# Use rocprof to capture .co
```

### Step 2: Disassemble to Editable Source

```bash
# Disassemble
llvm-objdump -d --mcpu=gfx950 reference.co > reference_disasm.s

# Convert disassembly to reassemblable source
# Key transformations needed:
# 1. Add .amdhsa_kernel block with correct KD fields
# 2. Add .amdgpu_metadata with .args section
# 3. Fix label references (branch targets)
# 4. Remove address prefixes from disassembly format
```

### Step 3: Reassemble and Verify Byte-Identical

```bash
# Reassemble
/opt/rocm/llvm/bin/clang -x assembler -target amdgcn-amd-amdhsa \
  -mcpu=gfx950 -o reassembled.co reference.s

# Verify byte-identical
llvm-objdump -d --mcpu=gfx950 reassembled.co > reassembled_disasm.s
diff reference_disasm.s reassembled_disasm.s
# Must show zero differences
```

**Critical**: If reassembly produces a different instruction count, the assembler silently dropped or changed instructions. Do NOT proceed until byte-identical reassembly is achieved.

### Step 4: Validate Numerical Equivalence

```bash
# Run reference binary
python3 bench.py --co reference.co --shape 256x256x4096
# cos_sim must be >= 0.9999 vs PyTorch reference

# Run reassembled binary
python3 bench.py --co reassembled.co --shape 256x256x4096
# cos_sim must match reference binary exactly
```

### Step 5: Iterate One Change at a Time

1. Make ONE edit to the .s file
2. Reassemble
3. Run validation
4. If cos_sim drops: revert and investigate
5. If cos_sim holds: commit and make next edit

**Never make multiple changes between validation runs.** The 28% speedup cautionary tale (Section 9) demonstrates why.

---

## 5. Validation Harness Patterns

### Standard GEMM Validation Harness

```python
import torch
import ctypes
import struct
import numpy as np

def load_kernel(co_path, kernel_name):
    """Load a .co binary and return launch function."""
    hip = ctypes.CDLL("libamdhip64.so")
    module = ctypes.c_void_p()
    hip.hipModuleLoad(ctypes.byref(module), co_path.encode())
    func = ctypes.c_void_p()
    hip.hipModuleGetFunction(ctypes.byref(func), module, kernel_name.encode())
    return hip, func

def cos_sim(a, b):
    """Cosine similarity between two tensors."""
    a_flat = a.flatten().float()
    b_flat = b.flatten().float()
    return (torch.dot(a_flat, b_flat) / 
            (torch.norm(a_flat) * torch.norm(b_flat))).item()

def validate_gemm(co_path, kernel_name, M, N, K, dtype=torch.bfloat16):
    """Validate GEMM kernel against PyTorch reference."""
    A = torch.randn(M, K, dtype=dtype, device='cuda')
    B = torch.randn(K, N, dtype=dtype, device='cuda')
    C = torch.zeros(M, N, dtype=dtype, device='cuda')
    
    # PyTorch reference
    ref = torch.mm(A.float(), B.float()).to(dtype)
    
    # Pack kernargs (must match kernel's expected layout exactly)
    kernargs = struct.pack('QQQ', A.data_ptr(), B.data_ptr(), C.data_ptr())
    # Add M, N, K, strides as needed by specific kernel
    
    # Launch kernel
    # ... (hipModuleLaunchKernel)
    
    sim = cos_sim(ref, C)
    max_diff = (ref.float() - C.float()).abs().max().item()
    print(f"Shape {M}x{N}x{K}: cos_sim={sim:.6f}, max_diff={max_diff:.6f}")
    return sim >= 0.9999

# Multi-shape validation
shapes = [
    (128, 128, 256),    # Single tile
    (256, 256, 4096),   # Standard
    (512, 512, 8192),   # Large
    (1, 4096, 4096),    # GEMV-like
    (4096, 1, 4096),    # Tall-skinny
]
for M, N, K in shapes:
    validate_gemm("kernel.co", "kernel_name", M, N, K)
```

### Attention Kernel Validation

```python
def validate_attention(co_path, B, S, H, D, causal=False):
    """Validate attention kernel against PyTorch reference."""
    Q = torch.randn(B, H, S, D, dtype=torch.bfloat16, device='cuda')
    K = torch.randn(B, H, S, D, dtype=torch.bfloat16, device='cuda')
    V = torch.randn(B, H, S, D, dtype=torch.bfloat16, device='cuda')
    
    # PyTorch reference with online softmax
    scale = 1.0 / (D ** 0.5)
    scores = torch.matmul(Q.float(), K.float().transpose(-2, -1)) * scale
    if causal:
        mask = torch.triu(torch.ones(S, S, device='cuda'), diagonal=1).bool()
        scores.masked_fill_(mask, float('-inf'))
    attn = torch.softmax(scores, dim=-1)
    ref = torch.matmul(attn, V.float()).to(torch.bfloat16)
    
    # Launch kernel and compare
    # ...
    
    sim = cos_sim(ref, output)
    print(f"B={B} S={S} H={H} D={D}: cos_sim={sim:.6f}")
```

### Multi-Configuration Sweep

```python
def sweep_configs(co_path, kernel_name):
    """Sweep across multiple configurations to find shape-dependent bugs."""
    configs = [
        # (M, N, K) - or (B, S, H, D) for attention
        # Start small, increase one dimension at a time
        (64, 64, 64),      # Minimal
        (128, 128, 128),    # Single tile
        (128, 128, 256),    # 2 K-iterations
        (128, 128, 4096),   # Full K-loop
        (256, 256, 4096),   # Multi-tile M,N
        (512, 512, 8192),   # Large
    ]
    failures = []
    for config in configs:
        sim = validate(*config)
        if sim < 0.9999:
            failures.append((config, sim))
    
    if failures:
        print("\nFailed configurations:")
        for config, sim in failures:
            print(f"  {config}: cos_sim={sim:.6f}")
        # Pattern analysis
        if all(c[2] > 256 for c, _ in failures):
            print("-> Bug manifests only with multiple K-iterations (K-loop bug)")
        if all(c[0] > 128 for c, _ in failures):
            print("-> Bug manifests only with multiple M-tiles (grid/addressing bug)")
```

### Benchmark Bias Warning

```python
# WRONG: zero_() creates unrealistic cache/memory state
A = torch.zeros(M, K, dtype=torch.bfloat16, device='cuda')
# bench.py using zero_() will show inflated TFLOPS because:
# 1. Zero data compresses in memory hierarchy
# 2. FP multiply by zero is fast-pathed
# 3. Cache behavior is unrealistic

# RIGHT: random data for benchmarking
A = torch.randn(M, K, dtype=torch.bfloat16, device='cuda')
```

---

## 6. Shape-Dependent Bug Manifestation

Some bugs only appear at specific shapes. This table maps common bugs to the shapes where they manifest.

### Shape-Dependent Bug Table

| Bug | Small (S<=128, M<=128) | Medium (S=1024, M=256) | Large (S>=8192, M>=4096) | Why |
|-----|----------------------|----------------------|------------------------|-----|
| Register clobber (ds_read into live VGPR) | May work | May work | **Fails** | More K-iterations increase probability of clobber; small K may not trigger the clobbered register path |
| Softmax accumulation error | Passes | **Marginal** (cos~0.999) | **Fails** (cos<0.99) | Longer sequences accumulate more FP error; online softmax rescaling amplifies |
| Grid addressing bug (wrong M/N tile) | Passes (1 tile) | **Fails** | **Fails** | Single tile has no grid; multi-tile exposes indexing errors |
| FP16 intermediate overflow | Passes | Passes | **Fails** | Larger reductions produce larger intermediate values |
| LDS bank conflict (performance only) | Not visible | Minor | **Significant** | Larger tiles use more LDS; conflict patterns emerge with wider access strides |
| Inter-WG coherence (barrier bug) | Passes (1 WG) | **May fail** | **Fails** | More workgroups increase race window |
| K-loop off-by-one | cos~0.9999 (1 step missed of many) | cos~0.99 | cos~0.5 | Missing 1 of 2 K-steps = half error; missing 1 of 128 K-steps = tiny error |
| Persistent kernel non-determinism | Not visible (1 tile) | cos~0.997 | cos~0.993 | More tiles = more scheduling variation |
| vmcnt FIFO ordering | May work (few loads) | **Fails** | **Fails** | More in-flight loads make FIFO position errors likely |
| accum_offset aliasing | **Fails** (always) | **Fails** | **Fails** | Not shape-dependent; structural KD bug |

### Testing Strategy Based on This Table

1. **If a bug appears only at large shapes**: suspect K-loop, accumulation, or FIFO ordering bugs
2. **If a bug appears at all shapes**: suspect structural bugs (KD, register mapping, MFMA layout)
3. **If a bug appears only at multi-tile shapes**: suspect grid addressing or inter-WG synchronization
4. **Always test at minimum 3 shape regimes**: single-tile, multi-tile, and production-scale

---

## 7. Performance Debugging

### In-Kernel Timing with s_memrealtime

```asm
; Read start timestamp
s_memrealtime s[N:N+1]       ; 64-bit timestamp into SGPR pair
s_waitcnt lgkmcnt(0)         ; wait for memrealtime to complete

; ... code to time ...

; Read end timestamp  
s_memrealtime s[N+2:N+3]
s_waitcnt lgkmcnt(0)

; Compute delta (store to output buffer for host readback)
s_sub_u32 s[N], s[N+2], s[N]
s_subb_u32 s[N+1], s[N+3], s[N+1]
```

**Note**: `hwreg(29)` (shader cycle counter) returns 0 on gfx950. Use `s_memrealtime` exclusively.

**Note**: `s_memrealtime` returns nanoseconds at memory clock frequency, not shader clock. Convert:
```python
shader_cycles = ns_delta * (shader_clock_mhz / 1000)
```

### hipEvent Profiling Overhead

```python
# hipEvent timing adds ~29ms overhead per measurement
# For short kernels, this dominates
# Use s_memrealtime for sub-microsecond measurements
# Use hipEvent only for kernels > 1ms

start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
start.record()
# launch kernel
end.record()
torch.cuda.synchronize()
elapsed_ms = start.elapsed_time(end)  # Includes ~29ms overhead for first call
```

### Roofline Analysis

```python
# MI355X hardware constants (see Appendix D for full table)
PEAK_BF16_TFLOPS = 1966.08  # BF16 MFMA peak (theoretical)
PEAK_FP8_TFLOPS = 3932.16   # FP8 MFMA peak (theoretical)
HBM_BW_TB_s = 8.0           # HBM3E bandwidth in TB/s
SHADER_CLOCK_MHZ = 1920     # Shader engine clock

# GEMM arithmetic intensity
flops = 2 * M * N * K
bytes_loaded = (M * K + K * N) * bytes_per_element  # A + B
arith_intensity = flops / bytes_loaded

# Roofline bound
memory_bound_tflops = HBM_BW_TB_s * arith_intensity / 1e12
compute_bound_tflops = PEAK_BF16_TFLOPS
roofline_tflops = min(memory_bound_tflops, compute_bound_tflops)

# Achieved efficiency
achieved_tflops = flops / (elapsed_seconds * 1e12)
efficiency = achieved_tflops / roofline_tflops
print(f"Achieved: {achieved_tflops:.1f} TFLOPS ({efficiency*100:.1f}% of roofline)")
```

### TFLOPS Calculation

```python
# GEMM
tflops = (2 * M * N * K) / (elapsed_us * 1e6)

# Attention (FWD)
# GEMM0: Q @ K^T, GEMM1: attn @ V
tflops = (2 * B * H * S * S * D + 2 * B * H * S * D * S) / (elapsed_us * 1e6)
# Simplified: 4 * B * H * S^2 * D / (elapsed_us * 1e6)

# FP8 grouped GEMM
# Sum across all groups
total_flops = sum(2 * M_i * N_i * K_i for M_i, N_i, K_i in groups)
tflops = total_flops / (elapsed_us * 1e6)
```

### Performance Counter Interpretation

Key counters from `rocprof`:

| Counter | What it Measures | Bad Value |
|---------|-----------------|-----------|
| SQ_WAIT_INST_LDS | Cycles stalled on LDS | > 30% of total cycles |
| SQ_INSTS_MFMA | MFMA instructions executed | Should match expected count from K-loop |
| SQ_WAIT_INST_VALU | Cycles waiting for VALU | High = missing NOPs causing replay |
| TCC_HIT_sum | L2 cache hits | Low = poor data reuse |
| FETCH_SIZE | Bytes fetched from HBM | Compare against theoretical minimum |

### Structural Performance Ceiling

Some performance limitations are structural and cannot be optimized away:

1. **MFMA utilization ceiling**: For grouped GEMM with small per-group K, each group pays MFMA drain cost (4x s_nop 15 = ~240 cycles). With K=256 and 16 groups, drain overhead is ~16%.
2. **LDS bandwidth ceiling**: If every MFMA needs LDS-sourced operands, LDS bandwidth (not compute) is the bottleneck.
3. **Memory-bound regime**: For small M,N with large K (e.g., GEMV), HBM bandwidth limits throughput regardless of compute optimization.

---

## 8. Top 10 Bugs by Frequency

Ranked by how often these bugs appear in gfx950 assembly development (GEMM, attention, grouped GEMM).

### #1: Register Clobber (Live VGPR Overwritten)

**Frequency**: Appears in virtually every kernel development effort. Multiple instances per project.

**Pattern**: A `ds_read_b128` or `buffer_load_dwordx4` writes to VGPRs that are still live (holding MFMA sources, scale factors, loop-invariant values, or accumulator aliases).

**Examples**:
- v[138:139] holding FP8 scale factor clobbered by ds_read_b128 v[136:139] (cos_sim dropped to 0.71)
- v[160:161] outer loop K offset clobbered by ds_read pair (wrong results only for K > tile_K)
- v[8:23] used as scratch temps overlapping live MFMA output VGPRs (intermittent wrong results)
- v[220:223] as buffer_load destination overlapping accumulator read path (non-deterministic)
- v[146:147] A-operand clobbered during MFMA instruction reordering

**Prevention**: Maintain a register liveness map. Before assigning any load destination, verify the target VGPRs have no live values. Color-code in comments:
```asm
; LIVE: v[0:7]=MFMA_A, v[8:15]=MFMA_B, v[16:31]=accum(alias AGPR), 
;       v[138:139]=scale
; DEAD: v[140:143], v[144:147]
ds_read_b128 v[140:143], v[addr]   ; OK: v[140:143] is dead
```

### #2: accum_offset / AGPR Allocation Errors

**Frequency**: At least once per kernel, often the first showstopper bug (multi-day debugging).

**Pattern**: The kernel descriptor's `accum_offset` field doesn't match the assembly's AGPR usage. Subtypes:
- `next_free_vgpr == accum_offset` -> zero AGPRs allocated -> MFMA output goes nowhere
- `accum_offset` too large -> VGPRs above it alias AGPRs -> ds_read clobbers accumulators
- `accum_offset` too small -> insufficient VGPRs for scratch/addressing

**Diagnostic**: Pre-fill C with -1.0. If -1.0 survives, zero AGPRs. If output is zero but not -1.0, AGPR aliasing.

**Fix**: 
```
.amdhsa_next_free_vgpr (accum_offset + total_agprs)
.amdhsa_accum_offset (first_agpr_number)
```
On gfx950, accum_offset is in bits [3:0] of rsrc3, encoded as (value/4 - 1).

### #3: vmcnt/lgkmcnt FIFO Ordering Errors

**Frequency**: Appears in every kernel with software pipelining or double-buffering.

**Pattern**: Programmer assumes waitcnt drains the "most recent" load, but hardware drains oldest first (FIFO). Prefetch loads issued before data loads invert the expected drain order.

**Example**:
```asm
; WRONG mental model:
global_load_dwordx4 v[0:3], ...   ; prefetch (issued first = position 0)
global_load_dwordx4 v[4:7], ...   ; data     (issued second = position 1)
s_waitcnt vmcnt(0)                ; Drains BOTH loads (correct but wasteful)
; Programmer thinks vmcnt(1) would drain the data load; actually it drains prefetch
```

**Fix**: Issue data loads BEFORE prefetches, or explicitly track FIFO positions:
```asm
; CORRECT: data first, then prefetch
global_load_dwordx4 v[4:7], ...   ; data     (position 0, drained first)
global_load_dwordx4 v[0:3], ...   ; prefetch (position 1, drained second)
s_waitcnt vmcnt(1)                ; Drains data load (position 0), prefetch stays in flight
```

### #4: Missing s_waitcnt vmcnt(0) Before s_endpgm

**Frequency**: Every single kernel needs this. Missing it causes stores to silently not land.

**Pattern**: `global_store` or `buffer_store` is issued, then `s_endpgm` executes before the store completes. The hardware retires the wave and the store is lost.

**Fix**:
```asm
; MANDATORY at every kernel exit point
s_waitcnt vmcnt(0)
s_endpgm
```

### #5: MFMA Column-Major Output Confusion

**Frequency**: Independently rediscovered by nearly every developer working on gfx950 MFMA.

**Pattern**: Programmer assumes MFMA output is row-major (a[k] = C[lane_row, col_group*4+k]). Actually column-major:
```
a[k] = C[col_group*4+k, lane%16]
```
Where col_group = which group of 4 VGPRs (VGPR_index / 4), and lane%16 = row.

**Consequence**: Store path transposes the output, producing a transposed result. For square matrices, cos_sim may be high (especially with symmetric data), masking the bug.

**Diagnostic**: Identity matrix test immediately reveals transpose.

### #6: flat vs global Memory Type Mismatch

**Frequency**: Appears in multi-kernel pipelines (attention, grouped GEMM with multiple passes).

**Pattern**: Kernel A writes with `global_store` (uses global address space, L1 coherence domain). Kernel B reads with `flat_load` (uses flat address space, different coherence domain). L1 cache serves stale data.

**Fix**: Use the same memory type throughout a kernel pipeline. Prefer `global_*` for all global memory operations. If mixing is unavoidable, use appropriate cache coherence flags (`sc0`, `sc1`, `nt`).

### #7: buffer_load WAW Hazard

**Frequency**: Appears in every kernel that uses buffer_load for MFMA operand prefetch.

**Pattern**: `buffer_load` writes to destination VGPRs asynchronously (on arrival, not on waitcnt). If the destination VGPRs hold live data, that data is clobbered at an unpredictable time.

**Key insight**: Unlike `global_load`, `buffer_load` has no interlock. The data arrives and overwrites the destination register immediately, regardless of whether you've issued a waitcnt.

**Fix**: Only buffer_load into registers that are dead (no longer needed) at the time of issue. Never reuse buffer_load destination registers for other purposes between issue and waitcnt.

### #8: NOP Hazard Violations

**Frequency**: Every kernel needs NOP management. Missing NOPs cause silent wrong results (not crashes).

**NOP Cheat Sheet**:
| Hazard | Required NOPs | Consequence of Violation |
|--------|--------------|------------------------|
| VALU -> MFMA (src dependency) | 2 NOPs (or 2 independent instructions) | MFMA reads stale VALU result |
| MFMA -> v_accvgpr_read | 4x s_nop 15 (total 64 NOPs) | Reads incomplete MFMA result |
| Transcendental -> read result | s_nop 3 | Reads stale value (no HW interlock) |
| v_readlane_b32 after VALU write | s_nop 1 | Returns stale lane value |
| s_waitcnt -> MFMA | 0 NOPs | (No hazard; waitcnt fully interlocks) |
| MFMA -> MFMA (same AGPR) | 0 NOPs (HW interlocked) | (No hazard) |
| ds_read -> MFMA (after waitcnt) | 0 NOPs | (No hazard; waitcnt handles it) |

### #9: .args Metadata Missing / KD Encoding Errors

**Frequency**: Every new kernel needs this right; typically hit once during initial bring-up.

**Pattern**: ROCm 7.2 on gfx950 requires a `.args:` section in the AMDGPU metadata YAML. Without it, `hipModuleLoad` succeeds but `hipModuleLaunchKernel` returns error 701.

**Minimum viable .args**:
```yaml
.args:
  - .offset: 0
    .size: 8
    .value_kind: global_buffer
```

**Other KD encoding bugs**:
- LDS size granularity: 128 dwords (512 bytes), not 64 dwords
- SGPR count: round up to allocation granularity (8 on gfx950)
- VGPR count: encoded as (count/4 - 1) on gfx950; off-by-one = crash or wrong results

### #10: s_movk_i32 Sign Extension

**Frequency**: Appears whenever a kernel needs an immediate value >= 32768 (0x8000).

**Pattern**: `s_movk_i32` sign-extends its 16-bit immediate to 32 bits. Loading 32768 (0x8000) produces -32768 (0xFFFF8000), not 32768.

**Fix**: Use `s_mov_b32` with a 32-bit literal for values >= 0x8000:
```asm
; WRONG
s_movk_i32 s0, 0x8000    ; s0 = 0xFFFF8000 = -32768

; CORRECT
s_mov_b32 s0, 0x8000     ; s0 = 0x00008000 = 32768
```

---

## 9. "Measuring Broken Code": The 28% Speedup Cautionary Tale

This is the single most important lesson from gfx950 assembly work. Read it. Internalize it.

### What Happened

During BWD attention kernel optimization, a kernel was reassembled and measured a 28% speedup. Then someone checked correctness.

### The Investigation

The reassembled kernel had **608 fewer instructions** than the reference. The assembler had silently dropped instructions during reassembly -- likely due to a syntax error that was treated as a comment, or a macro expansion failure, or `.set` directives inside the `.amdhsa_kernel` block that produced a stub.

The kernel ran. It produced output. It was fast. But the output was wrong.

### Why It Matters

1. **Broken code is always fast.** If you skip half the computation, you run in half the time. A 28% speedup from removing 608 instructions is not an optimization -- it's a bug.

2. **The GPU does not crash on wrong results.** Unlike CPU code where a null pointer dereference gives you a segfault, GPU code with wrong register mappings, missing NOPs, or dropped instructions simply produces wrong numbers. The kernel completes. The timing is valid. The output is garbage.

3. **Performance without correctness is meaningless.** Every performance measurement MUST be accompanied by a correctness check. No exceptions. Not even "I only changed the scheduling." Not even "I only removed NOPs." Not even "I only reordered loads."

### The Rule

```
BEFORE reporting any performance number:
1. Run correctness validation (cos_sim >= threshold)
2. If cos_sim < threshold, the performance number is INVALID
3. Fix correctness FIRST, THEN measure performance
4. Never trust a speedup you can't explain mechanistically
```

### How to Avoid This

1. **Disassemble after reassembly**: Always `llvm-objdump -d` your .co and verify instruction count matches
2. **Diff before and after**: `diff before.s after.s` should show only your intended changes
3. **Validate before benchmarking**: Run cos_sim check before running performance measurement
4. **Be suspicious of large speedups**: Anything > 5% from a single change deserves scrutiny. Anything > 10% almost certainly indicates a bug.
5. **Check for common assembler failure modes**:
   - `.set` directives inside `.amdhsa_kernel` block
   - Missing labels that are silently ignored
   - Macro definitions that shadow instruction mnemonics
   - Preprocessor directives consumed as comments

### Related: bench.py zero_() Measurement Bias

A separate but related trap: using `torch.zeros()` for benchmark input data. Zero data compresses in the memory hierarchy and takes fast paths through FP multiply units. A kernel that runs at 100 TFLOPS on zeros may run at 60 TFLOPS on random data. Always benchmark with `torch.randn()`.

---

## Appendix A: Quick Reference -- Register Allocation Checklist

Before launching any gfx950 kernel, verify:

- [ ] `next_free_vgpr` = `accum_offset` + `n_agprs` (not just `n_vgprs`)
- [ ] No VGPR above `accum_offset` is used for non-AGPR purposes (they alias AGPRs)
- [ ] All `ds_read_b128` destinations are in dead VGPR ranges
- [ ] All `buffer_load_dwordx4` destinations are dead at time of issue (WAW hazard)
- [ ] Scale factors / loop-invariant values are not in clobberable VGPR ranges
- [ ] SGPR buffer descriptors (s[4:7] patterns) are not clobbered between tiles
- [ ] VGPR pairs for 64-bit values are aligned to even register numbers
- [ ] SGPRs for scalar loads (s_load_dwordx4) are aligned to 4-register boundaries
- [ ] `.args` metadata section exists in AMDGPU metadata YAML

## Appendix B: Quick Reference -- Waitcnt Rules

```
Global loads:     vmcnt tracks; drains oldest first (FIFO)
Buffer loads:     vmcnt tracks; writes on arrival (WAW hazard!)
Global stores:    vmcnt tracks; must drain before s_endpgm
Buffer stores:    vmcnt tracks; must drain before s_endpgm
LDS reads:        lgkmcnt tracks; drains oldest first (FIFO)
LDS writes:       lgkmcnt tracks; must drain before s_barrier
Scalar loads:     lgkmcnt tracks
s_memrealtime:    lgkmcnt tracks

Key rules:
1. s_waitcnt vmcnt(0) before s_endpgm (ALWAYS)
2. s_waitcnt lgkmcnt(0) before s_barrier (ALWAYS)
3. Issue data loads before prefetches to control FIFO drain order
4. buffer_load destinations are clobbered on arrival, not on waitcnt
5. s_waitcnt before MFMA needs 0 NOPs (fully interlocked)
```

## Appendix C: Quick Reference -- MFMA Instruction Latencies


| Instruction | Cycles | AGPRs | Operand Size |
|-------------|--------|-------|-------------|
| v_mfma_f32_16x16x16_bf16 | 32 | 4 | 2xVGPR A, 2xVGPR B |
| v_mfma_f32_16x16x32_bf16 | 32 | 4 | 4xVGPR A, 4xVGPR B |
| v_mfma_f32_32x32x16_bf16 | 64 | 16 | 4xVGPR A, 4xVGPR B |
| v_mfma_f32_16x16x128_f8f6f4 | 48 | 4 | 8xVGPR A, 8xVGPR B |
| v_mfma_f32_32x32x64_f8f6f4 | 64 | 16 | 8xVGPR A, 8xVGPR B |

Co-execution window: Non-MFMA instructions can execute during MFMA latency. Plan ds_read, buffer_load, address computation to overlap with MFMA execution.

## Appendix D: Quick Reference -- MI355X Hardware Constants


| Parameter | Value |
|-----------|-------|
| CUs | 304 |
| Waves per CU | 4 (at 256 VGPRs), 8 (at 128 VGPRs) |
| VGPRs per CU | 512 (architectural), up to 256 per wave at occupancy 2 |
| AGPRs per CU | Shared with VGPR file via accum_offset |
| SGPRs per wave | 106 (102 user + 4 system) |
| LDS per CU | 128 KB (gfx950 has 64-bank LDS, not 32) |
| L1 cache per CU | 32 KB |
| L2 cache total | 96 MB |
| HBM3E bandwidth | 8 TB/s |
| Shader clock | 1920 MHz |
| BF16 MFMA peak | 1966.08 TFLOPS |
| FP8 MFMA peak | 3932.16 TFLOPS |
| Threads per wave | 64 |
| Max waves per workgroup | 16 |
