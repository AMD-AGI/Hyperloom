# LDS Optimization (gfx942 / gfx950)

> Distilled from `.claude/skills/lds-optimization/SKILL.md`. Diagnose
> bank conflicts and write-read latency stalls from rocprofv3 ATT traces, then
> apply swizzle / padding / write-read distance fixes.

## Hardware Quick Reference

| Aspect | gfx942 (MI300X) | gfx950 (MI350) |
|---|---|---|
| LDS per CU | 64 KB | 160 KB |
| Banks | 32 × 4 B | 64 × 4 B |
| Bank index | `(addr/4) % 32` | `(addr/4) % 64` |
| Peak throughput | 128 B/cyc | 256 B/cyc |
| Latency | ~20–40 cyc | ~2–64 cyc (bank-dependent) |
| Allocation granularity | 256 B | 1280 B (1280-byte aligned blocks) |
| Wavefront dispatch | — | 4-cycle waterfall over wave64 reads |
| MFMA transpose load | not available | `DS_READ_B64_TR_B{16,8,4}`, `DS_READ_B96_TR_B6` |
| VGPR file | arch_vgpr (separate from accum_vgpr) | same |

## LDS Op Async Model

```
ds_write_b32 v_addr, v_data    ; async; returns immediately
; ... independent work ...     ; the write completes in the background
s_waitcnt lgkmcnt(0)            ; stall until all LDS/SMEM ops drain
ds_read_b32 v_dest, v_addr     ; safe
```

Rules:
1. Any `ds_read` depending on a prior `ds_write` needs an `s_waitcnt
   lgkmcnt(0)` or `s_barrier` between them.
2. `s_barrier` is required for cross-wave write-then-read (lgkmcnt alone
   is not enough).
3. Longer write→wait distance = better latency hiding.

## Diagnose from ATT Trace

```python
import json
with open("ui_output/code.json") as f:
    inst = json.load(f)["code"]
# [ISA, _, LineNum, Source, Codeobj, Vaddr, Hit, Latency, Stall, Idle]
lds = [i for i in inst if i[0].startswith("ds_") or
                          ("lgkmcnt" in i[0] and i[8] > 0)]
print(f"LDS stall fraction: {sum(i[8] for i in lds) / sum(i[8] for i in inst):.1%}")
for i in sorted(lds, key=lambda x: x[8], reverse=True)[:15]:
    print(f"  L{i[2]:>4}  stall={i[8]:>6}  {i[0][:55]}")
```

### Type A — Bank conflicts (stall on the ds_* itself)
```
L766  stall=160  ds_read2_b64 v[44:47], v28 offset1:8
L767  stall=320  ds_read2_b64 v[36:39], v28 offset0:16 offset1:24
```
Sign: > 100 cycle stall per hit; `ds_read2_*` / `ds_write2_*` with offsets
that are multiples of the bank count (32 on gfx942, 64 on gfx950).

### Type B — Write-read latency exposed (stall on the `s_waitcnt` after a `ds_write`)
```
L761  stall= 960  ds_write2_b32 v28, v41, v43 offset0:32 offset1:48
L764  stall=4560  s_waitcnt lgkmcnt(0)        ; write didn't complete in time
L766  stall= 160  ds_read2_b64 v[44:47], v28
```
Sign: > 2000 cycle stall on the wait; few instructions between write and
wait.

### Type C — Cross-wave reduce serialization
```
L605  stall= 4080  s_waitcnt lgkmcnt(0)        ; wait for ds_bpermute
L606  stall=17024  s_barrier                    ; cross-wave sync
L607  stall=27220  s_waitcnt vmcnt(0)
```
Sign: pattern of `ds_bpermute → lgkmcnt → s_barrier → ds_write → lgkmcnt → s_barrier → ds_read`. Multiple barriers (> 4) in a reduce.

## Fix A — XOR Swizzle (zero overhead, preferred)

XORs row bits into column bits so that threads on the same column in
different rows hit different banks.

### Math
```
swizzled_col = original_col XOR (row >> shift)
```

### FlyDSL implementation
```python
allocator = SmemAllocator(None, arch="gfx942", global_sym_name="smem0")
lds_key = allocator.allocate_array(T.f16, KV_BLOCK_SIZE * HEAD_SIZE)

@flyc.kernel
def k(...):
    lds_base = allocator.get_base()
    lds_key_ptr = lds_key(lds_base)

    XOR_BITS = 4    # for fp16 vec=8, covers 4 banks per vec
    swizzled_col = col ^ ((row & 0x7) << XOR_BITS)
    lds_offset = row * HEAD_SIZE + swizzled_col
    lds_key_ptr.store(data, [lds_offset])
    # READ MUST USE THE SAME SWIZZLE
    val = lds_key_ptr.load([row * HEAD_SIZE + (col ^ ((row & 0x7) << XOR_BITS))])
```

### XOR16 swizzle (preshuffle GEMM convention)
```python
def swizzle_xor16(row, col, k_blocks16):
    rem = row % k_blocks16
    return col ^ (rem * 16)
```
Apply on BOTH write and read in `lds_store_*` and `lds_load_*` helpers in
`kernels/mfma_preshuffle_pipeline.py`.

### Mask sizing
| Element | Vec width | Banks/vec |
|---|---|---|
| f32 (4 B) | 4 | 4 banks (16 B) |
| f16/bf16 (2 B) | 8 | 4 banks (16 B) |
| fp8 (1 B) | 16 | 4 banks (16 B) |

XOR mask:
- gfx942 (32 banks): `32 / (vec * elem_bytes / 4) - 1`
- gfx950 (64 banks): `64 / (vec * elem_bytes / 4) - 1` — wider mask may be needed

## Fix B — Padding

Adds extra unused elements per row to break stride alignment.

```python
PADDING = 1
PADDED_STRIDE = HEAD_SIZE + PADDING
lds_key = allocator.allocate_array(T.f16, KV_BLOCK_SIZE * PADDED_STRIDE)
lds_offset = row * PADDED_STRIDE + col
```

Minimum padding (worst case): `bank_count / element_size_bytes`. Usually 1–4
elements suffice. Costs extra LDS bytes.

### Swizzle vs padding
| Aspect | Swizzle | Padding |
|---|---|---|
| LDS overhead | 0 | extra B/row |
| Compute | +1 SALU per addr | none |
| Risk | wrong mask = silent conflict | over-budget = launch fail |
| Preferred | LDS near capacity | LDS has headroom |

On gfx950 the 160 KB LDS gives generous padding headroom; on gfx942 prefer
swizzle.

## Fix C — Increase Write-Read Distance

When `ds_write → s_waitcnt lgkmcnt(0)` is back-to-back, the ~30 cycle LDS
latency is fully exposed.

Insert independent work between write and wait:

```python
# BEFORE
lds_ptr.store(data, [off])
gpu.barrier()
result = lds_ptr.load([off])

# AFTER
lds_ptr.store(data, [off])                                # async
next_off = compute_next_offsets()                          # SALU/VALU
next_data = buffer_ops.buffer_load(rsrc, next_off, 4)     # global load (async too)
scale = buffer_ops.buffer_load(rsrc_scale, scale_off, 1)
gpu.barrier()                                              # by now write completed
result = lds_ptr.load([off])
```

Priority order for what to insert:
1. Global loads for next phase (async, hides 300+ cycles)
2. Address compute (SALU/VALU)
3. Independent MFMA chains (~64 cycle each)
4. Scalar argument loads (`s_load_dword`)

Avoid inserting:
- Operations dependent on the LDS write
- More LDS ops (would compete for LDS bandwidth)
- Anything that bumps VGPR pressure past budget

## CDNA4 Transpose Load (gfx950 only)

For MFMA operand prep, `DS_READ_B{64,96}_TR_B{16,8,4,6}` loads transposed
data from LDS → VGPR in one instruction, eliminating the explicit
`ds_write` + permuted `ds_read` for transposition.

| Instruction | Element | VGPRs | What |
|---|---|---|---|
| `DS_READ_B64_TR_B16` | 16-bit (fp16/bf16) | 2 | Each lane holds 4 consecutive M or N |
| `DS_READ_B64_TR_B8` | 8-bit (fp8/bf8) | 2 | First loads K=0..7,16..23,32..39,48..55; second the rest |
| `DS_READ_B64_TR_B4` | 4-bit (int4) | 2 | First K=0..15,32..47; second the rest |
| `DS_READ_B96_TR_B6` | 6-bit | 3 | No even-VGPR alignment requirement |

Constraints:
- EXEC mask must be all-1s
- LDS address aligned to data size
- 64-bit DS ops require even-aligned VGPRs (except B96_TR_B6)

Empirical semantics from kernel-agents/memory: for `DS_READ_B64_TR_B16`,
**a 16-lane tile**: `dest[LIT[i]] = src[(LIT/4 + 4i)][LIT%4]`. Works with V
row-major LDS (no V^T pre-transpose needed). Supersedes earlier asm-playbook
claims. See `reference_ds_read_b64_tr_b16_semantics.md`.

## Verification

After fixing:
1. **Correctness**: swizzle must be applied **consistently** to both write
   and read paths. Run tests.
2. **Re-profile** with `/kernel-trace-analysis`. Check:
   - `ds_read_*` / `ds_write_*` stall ↓
   - `s_waitcnt lgkmcnt(0)` stall after writes ↓
   - No new bank conflicts introduced
3. **LDS usage** under budget (64 KB on gfx942, 160 KB on gfx950; gfx950
   allocates in 1280-byte blocks)
4. **VGPR count** not bumped significantly (swizzle: +1–2 SALU; padding: 0)

## Common LDS Patterns in Paged Attention

| Pattern | Location | Typical Issue | Fix |
|---|---|---|---|
| K/V cache tile in LDS | QK/PV MFMA loop | Bank conflict from `stride=HEAD_SIZE` | XOR swizzle |
| Softmax reduce via LDS | `ds_write → barrier → ds_read` | Write-read latency exposed + too many barriers | Increase distance; replace with `ds_bpermute` chain |
| Cross-wave max/sum broadcast | `ds_write → barrier → ds_read` from different wave | Cross-wave sync overhead | Merge max+sum into single reduce pass |
| MFMA accumulator shuffle | `ds_write accum → barrier → ds_read permuted` | Bank conflict if accumulator layout misaligns | Swizzle or `ds_bpermute` for permutation |
