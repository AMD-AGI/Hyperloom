---
title: FlyDSL attention — LDS layout and transposed reads on gfx950
kind: language
lever: flydsl_attn_lds
gens: [gfx950]
dtypes: [bf16, fp8]
regimes: [prefill, backward, decode]
updated: 2026-10-26
sources:
  - ROCm/FlyDSL@dd8373349:python/flydsl/expr/rocdl/__init__.py:17
  - ROCm/FlyDSL@dd8373349:python/flydsl/expr/rocdl/cdna4.py:31
  - ROCm/FlyDSL@dd8373349:lib/Dialect/FlyROCDL/CDNA4/CopyAtom.cpp:146
  - ROCm/FlyDSL@dd8373349:kernels/attention/flash_attn_utils.py:92
  - ROCm/FlyDSL@dd8373349:kernels/attention/flash_attn_utils.py:2725
  - ROCm/FlyDSL@dd8373349:kernels/attention/swa_gfx950.py:233
---

# LDS layout and transposed reads in a FlyDSL attention kernel

> P·V contracts over kv, and so do dV = Pᵀ·dO and dK = dSᵀ·Q in the backward pass. With K/V/Q/dO stored
> `[row, d]` in LDS, one operand of each of these GEMMs is strided along the contraction and must be read
> transposed. Generic LDS facts — 64 banks, read phases, swizzle vs padding, write-to-read distance, DMA
> waits — are in `../gemm/flydsl_gemm_datapath.md` §2–§4 and §9, `../optimize/lds-optimization.md` and
> `../../../../hardware/mi350_lds.md`. This card adds what attention adds.
>
> Numbers marked *(field)* are measured values. Use them for direction and order of magnitude, not as targets.

## Route here when
- You are deciding how V (forward) or dO, Q, K, dS (backward) reach the MFMA transposed.
- A bank-conflict counter or `SQ_WAIT_INST_LDS` is high on an attention kernel.
- You are changing an LDS row stride, padding or swizzle phase.
- You are considering a register transpose, a write-side pre-transpose or a "wider" transposed read.

## 1. The gfx950 transposed reads
- The complete set: `ds_read_b64_tr_b16` (16-bit), `ds_read_b64_tr_b8` (fp8), `ds_read_b64_tr_b4` and
  `ds_read_b96_tr_b6`. `ds_read_b128_tr_b16`/`_b8` and `ds_read_b64_tr_b32` are invalid instructions on gfx950;
  `ds_load_tr16_b128` (like the other `ds_load_tr*` ops) is a gfx1250 instruction. gfx942 has no transposed DS read.
- The `offset:` immediate is 16-bit unsigned. EXEC must be all ones and the LDS address aligned to the
  data size; `ds_read_b96_tr_b6` does not need an even VGPR pair.
- An 8-element-per-lane bf16 MFMA operand takes two `tr_b16` reads; each lane receives 4 consecutive
  values per read.
- A transposed bf16 operand moves 8 B per lane per instruction against 16 B for a K-contiguous
  `ds_read_b128`. When waves split the q rows, every wave needs the whole K/V tile, so bytes per wave are
  fixed by the tile and the transposed-read count is a structural constant. Budget the layout around it;
  no instruction halves it.

## 2. FlyDSL v0.3.2 API
`flydsl.expr.rocdl` re-exports the ROCDL ops, so three forms exist (`buffer_ops` is the repo's
`kernels/common/buffer_ops.py`, not part of the wheel):
```python
ptr = buffer_ops.create_llvm_ptr(fx.Int64(byte_off), address_space=3)    # !llvm.ptr<3>
v4 = rocdl.ds_read_tr16_b64(v4f16_ty, ptr).result      # op class: take .result
v4 = rocdl.ds_read_tr16_b64_(v4f16_ty, ptr)            # function form
atom = fx.make_copy_atom(rocdl.cdna4.LDSReadTrans16_64b(), bf16)   # copy atom
```
- The other widths: `ds_read_tr8_b64`, `ds_read_tr4_b64`, `ds_read_tr6_b96`; atoms `LDSReadTrans8_64b`,
  `LDSReadTrans4_64b`, `LDSReadTrans6_96b`, or `rocdl.cdna4.LDSReadTrans(granularity, bits)` with only
  (16, 64), (8, 64), (4, 64), (6, 96) accepted.
- With a base pointer plus a compile-time constant, the op folds the constant into `offset:`; the ISA is
  identical to inline asm with an immediate.
- **Use the op or the atom, not inline asm,** unless the reads sit in a hand-scheduled asm region that supplies its
  own waits (`../gemm/flydsl_gemm_datapath.md` §4). The compiler does not track an asm read's destination, so
  every consumer needs a hand-written `lgkmcnt` drain, pinned with `sched_barrier(0)`, which cuts the
  P·V cluster into fragments (`../gemm/flydsl_gemm_datapath.md` §4). *(field)* Swapping asm for the op
  emitted the same 96 reads, let the backend place graded waits, cut the slowdown from split K/V reads
  from 3.26% to 1.95%, dropped spills at a 128-register target from 14–105 to 2, and measured occupancy rose
  from 2.985 to 3.978 waves/SIMD.
- FlyDSL's dual-wave forward reads V with inline asm and immediate offsets
  (`_ds_read_tr16_b64_imm`); its generic and SWA paths use the op and the atom.

## 3. Lane maps and k order
- **tr_b16** *(field: measured with a position-encoded LDS image)*: each 16-lane group is independent;
  lane L supplies the 4 consecutive 16-bit values `r[L][0..3]` at its address, and output lane m, slot j
  is `r[4j + m//4][m % 4]`. If lane L addresses row `L//4`, columns `4·(L%4)…+3` of a 4×16 block, lane m
  receives column m's 4 rows: `A[m = column][k = row]`.
- **tr_b8** *(field)*: lane j of a group supplies 8 consecutive bytes. With lane j addressing
  `(j//2)·W + (j%2)·8` (W = row bytes), the group transposes an 8-row × 16-byte block and lane j receives
  column j's 8 rows. Four reads give the 32 B/lane operand of a 16x16x128 fp8 MFMA.
- The lane→column map is fixed; the addresses choose only which rows and which 16-column chunk.
- **k order is free, but A and B must use the same one.** `C = Σ_k A[k][m]·B[k][n]`, so any row
  permutation the addresses produce is correct when both operands share it. *(field)* Switching only dS
  to `tr_b8` while K kept its old k order dropped SNR to −1.8 dB. Two `tr_b16` reads 16 rows apart give a
  lane kv `{b..b+3} ∪ {b+16..b+19}`; one `tr_b8` gives 8 consecutive rows. The sets differ, so no byte
  shuffle inside a lane reconciles them: change the write-side layout.
- Probe a read with an LDS image holding its own index and an operand constant along k. All-ones inputs
  cannot see a wrong lane map.

## 4. Transposed reads need whole 16-element blocks
- A swizzle phase in element bits 4–5 (`col ^ ((row & 7) << 4)` on a 128-element row) keeps the
  transposed-read family conflict-free by construction. A non-transposed read that walks 16 rows of one
  column on the same image gets only 4 phases: 4-way.
- **Never move a phase bit into element bit 3** (blocks smaller than 16 elements). *(field)* `(row&7)<<3`,
  and a split phase built to leave the transposed reads' bank coverage unchanged, both read: conflicts
  16.4% → 46.5%, `SQ_LDS_IDX_ACTIVE` +56%, wave cycles +9.6% / +10.4%, same `ds_read` count. The
  transposed reads were split into more passes; bank coverage was not the cost.
- To serve a second read family with a different phase, give it its own copy of the tile when LDS allows,
  instead of retuning the shared phase.
- Where a swizzle mask is combined with another address term by `+`, check the two bit fields stay
  disjoint under the new phase. *(field)* An overlap gave finite, silently wrong dK/dV (norms off 0.06% and
  0.48%). Gate layout arms on bitwise-equal output norms (`flydsl_attn_correctness.md`).

## 5. Bank-conflict counters are usually a red herring
- Compare `SQ_LDS_BANK_CONFLICT / SQ_LDS_IDX_ACTIVE` with a floor computed by hand from the access widths
  (`../gemm/flydsl_gemm_datapath.md` §3), then prove LDS bounds the kernel with a probe before touching
  the layout (`flydsl_attn_attribution.md`).
- *(field)* On MFMA-latency-bound backward kernels, conflicts 19% → 0% and 21.7% → 4.4% bought no speed.
- With bf16 `head_dim = 64` (128-byte rows) the row index adds almost no bank phase and a 64-element row
  leaves little room for an XOR phase: *(field)* every XOR and padding variant stayed near 19%. A
  128-element physical row with `(row&7)<<4` reached 0%; packing two logical rows into each 128-element
  row kept LDS size and global bytes unchanged.
- Separate a second workgroup's LDS pressure from layout conflicts with a forced-occupancy-1 probe (pad
  LDS with a dead buffer). *(field)* At occupancy 2, 7.6 points of a 22.8% conflict metric came from the
  second workgroup's port and timing pressure; no swizzle moves that part.
- When one tile is read both naturally (QK) and transposed (P·V), one pad cannot satisfy both read
  families; decide which family is on the critical path first.

## 6. Row stride: keep 16-byte alignment
- *(field)* A K/V padding sweep: strides whose rows were only 8-byte aligned split every `ds_read_b128`,
  3.97 ms against 1.84 ms (2.2× the time), with `SQ_LDS_BANK_CONFLICT` at 0 throughout. All 16-byte-aligned pads
  were healthy but not equal (−0.1% to −6.2% around the deployed value).
- Sweep the pad once on a new kernel, with an ISA gate on the `ds_read_b128` count; do not re-sweep a
  tuned one.
- With 16-byte alignment the dword stride is a multiple of 4, so padding alone cannot push a wide read
  below its conflict floor; only an XOR phase rotation with a matching writer can.

## 7. What does not cut transposed-read cost
- **A wider transposed read** does not exist (§1).
- **Write-side pre-transpose through LDS DMA.** One `buffer_load_dwordx4 … lds` chunk is 8 consecutive d
  of one kv; a kv-contiguous image needs 8 consecutive kv of one d, a 2-byte scatter. The free per-lane
  global offset only chooses which row a chunk lands in.
- **A register transpose** (`ds_bpermute` gathers, about 8 permutes and 12 selects per operand, replacing
  one `tr_b16` read) pays only when one transpose is reused across many tiles. *(field)* A backward
  kernel whose waves each own a d-tile and transpose Q once for all rank tiles: backward +11.8%. Per-tile
  transposes: forward P·V −77%; a dense backward 3.5× slower while its conflicts fell to 3%.
- **Fewer LDS instructions is not the currency.** *(field)* Collapsing transposed-read pairs into plain
  `ds_read_b128` where the layout allowed (512 reads removed, −124 instructions) ran 1.65–1.87% slower:
  `lgkmcnt(0)` waits 53 → 118. Count graded waits, not reads.
- **One asm block for a run of reads, drained once,** helps only where the compiler drains each read
  separately. *(field)* +0.6% there; 0.86–0.93× where the compiler already overlapped the reads.

## 8. Addresses
- **A runtime LDS address term loses the DS immediate offset**, and every access recomputes its address.
  *(field)* A per-access runtime XOR swizzle: −23%. A block index made runtime turned every `tr_b16` offset
  runtime: 825 → 551 TF/s; returning it as a compile-time int under `const_expr` restored 827.
- Keep swizzles as compile-time layout functions. Put per-slot or per-wave variation in the base pointer.
- Keep slot and tile offsets in bits the swizzle does not XOR. If the mask XORs the bits that also carry
  the d-tile index, address differences become lane-dependent and each transposed read needs its own
  address VGPR. *(field)* A second LDS slot raised the read-base registers from 59 to 96 and spilled 197
  dwords.

## 9. Keep LDS write bursts out of transposed-read bursts
*(field)* A legal restructuring that moved the next head's `ds_write` burst into the middle of a
transposed-read burst cost about 2.4% after crediting the barriers it removed. Dense segments are sensitive
to LDS port contention; schedule write bursts where the read family is idle.

## Verify
- The ISA shows only `ds_read_b64_tr_*` / `ds_read_b96_tr_b6` for transposed operands, issued by op or
  atom, with immediate offsets.
- Both operands of every MFMA use the same k order, checked with a position-encoded probe.
- Swizzle phases keep 16-element blocks for transposed reads; rows are 16-byte aligned.
- Any conflict-reduction claim comes with a hand-computed floor and a wall-time probe.

## Related
`flydsl_attn_mfma_layout.md` · `flydsl_attn_schedule_sync.md` · `flydsl_attn_registers_occupancy.md` ·
`flydsl_attn_attribution.md` · `flydsl_attn_correctness.md` · `../gemm/flydsl_gemm_datapath.md` ·
`../optimize/lds-optimization.md` · `../../../../hardware/mi350_lds.md`
