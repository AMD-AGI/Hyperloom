---
title: FlyDSL GEMM — LDS, global feed and prefetch on gfx950
kind: language
lever: flydsl_gemm_datapath
gens: [gfx950]
dtypes: [fp8, bf16, fp16, mxfp4]
regimes: [compute-bound, feed-bound]
updated: 2026-10-19
sources:
  - ROCm/FlyDSL@dd8373349:kernels/gemm/preshuffle_gemm.py:283
  - ROCm/FlyDSL@dd8373349:kernels/common/mma/mfma_preshuffle_pipeline.py:30
  - ROCm/FlyDSL@dd8373349:kernels/gemm/gemm_a16w16_gfx950.py:489
  - ROCm/FlyDSL@dd8373349:python/flydsl/expr/gpu.py:157
  - ROCm/FlyDSL@dd8373349:python/flydsl/expr/rocdl/universal.py:59
  - AMD-AGI/Primus-Turbo@7615a5a2:primus_turbo/flydsl/utils/gemm_helper.py
  - AMD-AGI/Primus-Turbo@7615a5a2:primus_turbo/flydsl/utils/prims.py
  - AMD-AGI/Primus-Turbo@7615a5a2:primus_turbo/flydsl/gemm/gemm_mxfp4_kernel.py
---

# LDS, global feed and prefetch in a FlyDSL GEMM

> Numbers marked *(field)* are measured values. Use them for direction and order of magnitude, not as
> targets. ISA statements were checked with `llvm-mc` / `llc -mcpu=gfx950`. "PS" is FlyDSL's
> `kernels/gemm/preshuffle_gemm.py`; "GH", "PR" and "FP4" are Primus-Turbo's `utils/gemm_helper.py`,
> `utils/prims.py` and `gemm/gemm_mxfp4_kernel.py` under `primus_turbo/flydsl/` at main `7615a5a2`. "A16" is
> FlyDSL's `kernels/gemm/gemm_a16w16_gfx950.py`.

## Route here when
- An LDS, bank-conflict, swizzle or `lgkmcnt` question in a GEMM.
- Deciding how an operand reaches the MFMA: LDS, direct-to-VGPR, sync copy or DMA.
- Global loads or C stores look like the cost, or a prefetch did not help.

FlyDSL's GEMM guide shows the baseline data path on `preshuffle_gemm`: A through LDS by sync copy or
async DMA, B preshuffled straight to VGPRs, the XOR swizzle and its write/read pair
(`../optimize/gemm-optimization.md` §3, §4, §11, §12). The general LDS and prefetch skills
(`../optimize/lds-optimization.md`, `../optimize/prefetch-data-load.md`) cover diagnosis from a trace,
swizzle vs padding, write-to-read distance and the prologue/loop/epilogue prefetch transformation, for any
kernel. Hardware numbers live in `../../../../hardware/mi350_lds.md`.

## 1. Allocate LDS the v0.3.2 way

Declare the buffers as an `@fx.struct` of `fx.Array` fields and allocate them in the kernel with
`fx.SharedAllocator().allocate(Storage).peek()` (PS:214, 283). With the default `static=True` each
field becomes its own LDS global and the backend places it (`expr/gpu.py:157`): field order is not
a layout you control, and a field an ablation arm stops using is pruned, which moves the others. An
epilogue that reuses the operand buffers declares them as an `@fx.union` (A16's `SharedStorage`
overlays `ab` and `c`). `SmemAllocator` / `SmemPtr` are legacy.

## 2. Swizzle at 16-byte granularity, composed into the layout

- XOR whole 16-byte slots, never elements inside a vector: `swizzle_xor16` is
  `col ^ ((row % k_blocks16) * 16)` with `k_blocks16 = tile_k·elem_bytes / 16`
  (`mfma_preshuffle_pipeline.py:30`). XOR-ing at element granularity scrambles the lanes of a
  `ds_read_b64/b128`.
- PS composes the swizzle into the LDS view instead of computing addresses:
  `fx.make_composed_layout(fx.static(fx.SwizzleType.get(bits, base, bits)), ordered_layout)`, with
  `bits = log2(k_blocks16)` and `base = log2(elements per 16 B)` for 8-bit or DMA-written tiles. The
  writer and the reader then agree by construction — and the DMA path, which computes addresses by
  hand, applies the same `k ^ ((m % k_blocks16)·elems_per_16b)` (PS:368).
- The swizzle is per-lane VALU (or folded into the layout), not "one SALU": the row index lives in a
  VGPR.
- Upstream measured the choice: on the bf16 sync path PS keeps a fixed `Swizzle<3,3,3>` because it
  ran 1011 TF/s against 417 TF/s for the derived swizzle at `tile_k=256` (comment at PS:288).
  Re-measure a swizzle change; "derived from the tile width" is not automatically better.
- **One function computes the key for both sides.** Turbo's `lds_row_swizzle(row, chunks) =
  ((row % 16) // 2) & (chunks - 1)` (in 16 B chunks) is called by the G2S writer (`swizzle_128`) and by
  the transpose reader, "so the two cannot drift" (GH:220-231).
- **Transpose reads need a wave term too.** With a 1024 B per-wave chunk stride, every wave's
  transpose reads hit the same banks. Turbo XORs the chunk index with `W << 1` on both sides (`wswz`,
  GH:1809-1830, 1863). *(field)* Bank conflicts on that kernel went from 14% to 0%.

## 3. Bank conflicts: compute the floor, then measure

**First tell the three LDS failure modes apart; each has a different fix.**

| Mode | How it shows up | Fix |
|---|---|---|
| Capacity | LDS per workgroup caps occupancy | smaller tile or fewer stages |
| Layout | bank conflicts from the stride or the access pattern | swizzle or padding |
| Timing | a `ds_write` too close to the `ds_read` that depends on it | more write-to-read distance (`../optimize/lds-optimization.md`) |

Treating every LDS symptom as a swizzle problem wastes the most time: check the write-to-read distance before
the layout.

gfx950 LDS has 64 banks of 4 bytes; bank = `(byte_addr / 4) mod 64`. For a 32-bit access whose lane
stride is `s` dwords, the lanes of one issue phase touch `64 / gcd(s, 64)` distinct banks. A 128-byte
row stride (`s = 32`) lands every lane on two banks — a heavy conflict, not the "2-way" an old note
claimed. Wider accesses (`ds_read_b64`, `b128`) serve fewer lanes per phase; do the same count per
phase.

- **Swizzle or pad.** XOR swizzle when the access pattern is regular and LDS headroom is tight; padding when
  a small stride bump removes the conflict and the swizzle arithmetic would clutter the address code.
  gfx950's 160 KiB makes padding affordable more often than on gfx942. A swizzle or padding derived for
  gfx942's 32 banks is unverified on gfx950; re-derive it.
- Padding must keep the 8- or 16-byte alignment a `ds_read_b64/b128` needs. `+1 element` on fp16 breaks
  it; pad by whole 16-byte slots, or swizzle.
- Compare `SQ_LDS_BANK_CONFLICT` against the floor you computed by hand, not against zero.
- **Zero conflicts is not a speedup and LDS is not the bound until a probe says so.** *(field)* A
  kernel with 81% of wave time in `WAIT_INST_LDS` gained 0.0% when its LDS reads were halved; the
  stall counter could not say whether LDS bounded it. Use a halve-the-reads probe
  (`flydsl_gemm_attribution.md`).

## 4. Waits around LDS

- **Same wave, write then read: no wait.** LDS operations of one wave complete in order;
  `llc -mcpu=gfx950` emits `ds_write_b32` directly followed by `ds_read_b32` of a neighbouring lane's
  slot.
- **Another wave reads what you wrote:** the writer waits `lgkmcnt(0)`, then everyone meets at
  `s_barrier` — which is what `fx.gpu.barrier()` lowers to.
- **A FLAT (generic-address) access counts in both `vmcnt` and `lgkmcnt`.** *(field)* Waiting
  `lgkmcnt(0)` after a flat load from global took 96 ticks against 56 after a `global_load`, and both
  took about 250 to `vmcnt(0)`. The lgkm count retires before the data arrives, so `lgkmcnt(0)` alone
  does not make a flat load's data ready. Keep hot-loop pointers in the buffer or global address space.
- `lgkmcnt` also counts scalar memory loads, which return out of order. A long stall on
  `s_waitcnt lgkmcnt(0)` may be an `s_load`, not LDS; keep scalar loads out of that window instead of issuing
  them there to overlap with the LDS wait.
- LDS data operands may be AGPRs on gfx950 (`ds_read_b32 a0, v1` and `ds_write_b128 v1, a[0:3]`
  assemble); only the address must be a VGPR.
- `lgkmcnt` is a 4-bit field (max 15) and `vmcnt` a 6-bit field (max 63).
- **After an intrinsic LDS DMA, the compiler drains everything before your `ds_read`.** It cannot prove
  which reads alias the DMA's destination, so it inserts `s_waitcnt vmcnt(0)`. Turbo issues those
  `ds_read`s in inline asm and supplies the wait itself (`vmcnt_hint`, `_packed_ds_read_tr_offsets`,
  GH:1783). *(field)* At long K this drain removal gave +2.3%; `vmcnt(3)` was the deepest setting that
  stayed bitwise repeatable. An asm `ds_read` must also drain its own `lgkmcnt` before the consumer,
  because the compiler does not track it. `cdna4.BufferLoadAsyncLDS*` with `rocdl.asyncmark()` /
  `wait_asyncmark(n)` is the in-DSL alternative (A16 uses it).

## 5. Global → LDS DMA

| | gfx950 | gfx942 |
|---|---|---|
| `buffer_load_dword … lds` | yes | yes |
| `buffer_load_dwordx4 … lds`, `global_load_lds_dwordx4` | yes (16 B/lane) | no |
| `ds_read_b64_tr_b16` and the other transpose reads | yes | no |

- FlyDSL atoms: `fx.rocdl.BufferCopyLDS128b` (PS's `dma_a_to_lds`; synchronous — the compiler inserts
  the `vmcnt` wait) and `fx.rocdl.cdna4.BufferLoadAsyncLDS128b` (A16; you close groups with
  `rocdl.asyncmark()` and wait with `rocdl.wait_asyncmark(n)`). The DMA is a VMEM load that writes LDS:
  it counts in `vmcnt` like any other load.
- The MUBUF immediate `offset` is 12 bits. `llvm-mc` encodes `offset:4096` as `offset:0` without an
  error (`0x1000` lands outside the field), so a hand-written `offset:4096` silently reads the wrong
  place. *(field)* On gfx950 the LDS-DMA immediate applies to the LDS destination as well as the
  memory address, which lets a wave-major fill share one M0 write across steps; it was worth ~0.9%
  per cell.
- The M0 base of an LDS DMA must never go negative; see `flydsl_gemm_correctness.md` §5.

**How Turbo issues the DMA** (`G2SLoader`, GH:305-386):
- One `fx.make_copy_atom(fx.rocdl.BufferCopyLDS128b(), 128)` per 16 B per lane; each wave writes 1 KiB per
  step.
- The per-lane global offsets are Python ints computed at trace time, so the hot loop has no address
  VALU. The wave-uniform K offset rides the SGPR `soffset`: `fx.copy(atom, src, dst, soffset=...)`.
- **M0.** In the whole-loop asm each fill is `s_add_u32 m0, <pool base>, <step·nwaves·1024>` followed by
  `buffer_load_dwordx4 … offen lds`. The MXFP4 kernel uses a **wave-major** layout instead: each wave's
  chunks are contiguous, so consecutive steps are 1 KiB apart and ride the 12-bit immediate. One M0 write
  then covers a 4 KiB window (`g2s_lds_imm`, GH:294; `_MXFP4_G2S_IMM`, FP4:583-586). The immediate applies
  to the memory address too, so pre-subtract it from the global voffset, which must stay ≥ 0.

## 6. What a global access costs

- **The cost unit is unique 128-byte lines touched per instruction, not bytes or requests.**
  *(field)* Narrowing a `buffer_load … lds` to a quarter of its width with the same lines touched
  bought 0.14%; cutting the lines one instruction touches from 8 to 1 at the same instruction count
  bought 20.2%. Storing an fp6 A operand densely (96-byte rows) saved 25% of the bytes but scattered
  each wave's DMA over 64 lines and lost 26.6%; re-tiling it so a wave's 64 lanes read 1 KiB
  contiguously turned the same byte saving into +2.0%. Merging requests pays only when it also cuts
  unique lines.
- **A contraction stride ≡ 64 mod 128 bytes straddles lines.** Padding K so the row pitch is a
  multiple of 128 B removed the straddle *(field: 2196 → 2554 TF/s)*.
- **Stores go out in 64-byte requests on gfx950.** Pairing two columns per lane halves the request
  count *(field: 2.36e7 → 1.21e7 write requests)*; `dwordx2` alone only cuts instruction count. Watch
  `TCP_TCC_WRITE_REQ` and `TA_BUFFER_WRITE_WAVEFRONTS` together.
- **Price the C store before optimizing it.** *(field)* An instruction-class ablation of one fused
  epilogue put 6.3% on the stores and 0.95% on the VALU; store cost scaled with 1/K, not with tile
  shape.
- **A "non-temporal" win may be coalescing.** *(field)* A ~10% gain first credited to the `nt` bit came
  from whole-row coalescing the same change introduced. Choose `nt` per load by reuse class (scale
  streams that are re-read stay cached).
- **LDS immediates have windows.** A base at or above `0x10000` cannot use the 16-bit DS offset
  (`ds_read_b128 … offset:65536` does not assemble), and `ds_read2st64`'s 8-bit offset (256-byte units)
  breaks when tiles widen. *(field)* Doubling the row stride from 512 to 1024 B pushed offsets out of
  that window and the paired reads split: +38% `ds_read` instructions.

## 7. Prefetch inside the K loop

- Carry the prefetched values through the loop: with `range(..., init=[...])` only yielded values
  persist; a plain `range()` carries every name the body reassigns
  (`../bottleneck/debug-flydsl-kernel.md` § The three loop forms). Carry everything the next
  iteration needs — scale values and addresses, not only the tiles.
- **Prefetch pays only when independent work can fill the latency it exposes**: next-tile loads, address
  arithmetic, MFMAs that do not depend on the load in flight, epilogue setup that touches no data in flight.
  The general transformation (prologue, loop-carried buffers, epilogue) is `../optimize/prefetch-data-load.md`.
- **Prefetch hides latency, not capacity.** *(field)* When the working set thrashed L2, deeper prefetch
  bought nothing; fix the tile order (`flydsl_gemm_schedule_and_locality.md` §2) or the footprint.
- **Bypassing LDS for an operand several waves share costs twice.** Each wave reloads it from global,
  and at one workgroup per CU nothing hides the longer L2 latency. *(field)* In an 8-wave kernel, B
  direct-to-VGPR ran ~22% slower; A direct-to-VGPR ran 2.5× slower on a long K. PS can load B straight
  to VGPR because B is pre-shuffled and each wave owns its own N slice.
- A prefetch that pushes the kernel over a register tier costs more than it hides; re-read
  `.vgpr_count` after adding one.
- **Gate a prefetched load exactly as its consumer is gated**, in the prologue and in the loop body. A
  load behind a mode flag (e.g. a scale load that only one quantization mode needs) that is gated
  differently in the two places feeds the consumer a value from the other branch.

## 8. Buffer descriptors: per-tile i64 rebase, pinned to SGPRs

The buffer descriptor's `num_records` and the per-lane `voffset` are 32-bit, so one descriptor addresses
at most 4 GiB.
- **Fold each tile's large base into the descriptor's 64-bit base** and keep only the in-tile offset in
  `voffset`/`soffset` (`make_fp8_buffer_tensor_rebased`, GH:118-147). Bound `num_records` to the rest of the
  tensor from that base: out-of-range M/N/K reads then return zero for free.
- **Pin base and `num_records` with `readfirstlane`.** A value the divergence analysis cannot prove
  uniform lands in VGPRs, and a descriptor built from VGPRs puts every buffer access behind a
  readfirstlane waterfall (`_readfirstlane_i32`, PR:338-344). *(field)* Pinning gained 13–16%.
- When one operand's strided traversal spans 2³² bytes or more, rebase the descriptor per load (the
  `rebase` mode of `G2SLoader`). Clamp the shifted `num_records` at 0 for tiles past the end.
- Pass tensors to the entry function at their full rank. A flat view of more than 2³¹ elements
  overflows the int32 shape pack.
- Check SNR on both sides of the 2³¹-element threshold. Equal SNR means there is no overflow break.

## 9. K-row operands: hardware transpose reads

- `ds_read_b64_tr_b8` (fp8) and `ds_read_b64_tr_b16` (bf16) read a K-row tile (stored `[K, ·]`) from LDS directly into MFMA
  operand order (gfx950 only). Turbo runs NN and TN natively with them, with no transposed copy of either
  operand (`S2RLoaderTr`, GH:1831-1963).
- Pack the four reads of one fragment onto one base VGPR with immediate offsets. One address VGPR
  instead of four avoids an address-register spill (`_packed_ds_read_tr_offsets`, GH:1783-1805).
- The lane→column map of a transpose read is fixed by the hardware. Only which 16-column chunk a 16-lane
  group reads can be chosen through the address.
- *(field)* Do not "halve the instruction count" by replacing four `ds_read_b64_tr_b8` with two `ds_read_b128`
  at the same addresses. The probe ran 80% slower: transpose-read base addresses are laid out per
  transpose unit, and `b128` at those addresses collides on banks. The instruction and the LDS layout must
  change together.
- *(field)* A transpose-read operand costs more LDS instructions per MAC than a K-contiguous one: 8 B per read
  against 16 B. An NT kernel's steady state ran 11–14% faster per K iteration than the same tile on TN.

## 10. Operand layout: preshuffle or permute in the fill; keep rows on cache lines

There are two ways to get an operand into MFMA order without a runtime shuffle:
- **Preshuffle it on the host.** FlyDSL's PS takes B as `[N/16, K/kpack, 4, 16, kpack_bytes]`, so each
  `buffer_load_dwordx4` lands in MFMA register layout and B skips LDS (`../optimize/gemm-optimization.md` §12);
  the MXFP4 kernels take `shuffle_weight_w4` / `shuffle_scale_w4` layouts. The cost is a host-side pass, paid
  once for weights that do not change, as in inference.
- **Permute rows in the fill.** Changing only which global row each G2S load fetches reorders a lane's
  n-fragments with no runtime cost, and the LDS image, swizzle and reads stay the same
  (`compute_global_swizzle_pair`, GH:256-267; epilogue use in `flydsl_gemm_epilogue.md` §3). Turbo's dense
  paths use this and need no offline step, which matters in training, where weights change every step.

- **Keep every operand row on a 128 B line.** Take the row stride from the tensor, not from K, so a
  caller can pad rows (`row_bytes`, FP4). For MXFP4, `K % 256 == 0` is what puts the row stride `K/2` on
  a 128 B line; it is not a kernel constraint. *(field)* With the block count fixed, row phases of 64,
  32 and 96 B cost +8.4%, +11.6% and +12.0%. Pad K when memory allows.

## 11. MX scales

- **Pack four E8M0 bytes per dword.** The scaled MFMA picks a byte with `op_sel` (`scale_opsel(k, pack) =
  k % pack`, GH:114). The field is 2 bits, so 4 is a hardware limit. *(field)* Going from 1 to 4 bytes per
  dword cut VMEM instructions by 14% and gained 5–8%.
- **A wrong `op_sel` silently scales by another K block's exponent.** Probes with `randn`-derived scales
  cannot see it, because all exponents are similar. Use random exponents in the correctness probe.
- **Scales can skip LDS.** Turbo's MXFP4 loads them with `buffer_load_dwordx4` straight into pinned VGPRs
  (FP4:1393-1417). The LDS path delayed consumption by the refill-ahead distance, so a direct load
  must offset its K position by that lag.
- **Do the scale preshuffle in the same host call.** One `@flyc.jit` host stub launches the preshuffle kernel
  and then the GEMM, so the scales are repacked in stream order with no extra Python dispatch
  (`build_preshuffle_ab_kernel`, GH:2115-2122, 2262). A caller that reuses weights can preshuffle their
  scales once (`preshuffle_mxfp4_scales`, FP4:4877).
- Never store E8M0 scales with a non-temporal hint (`flydsl_gemm_epilogue.md` §5).

## Related
`flydsl_gemm_mainloop.md` · `flydsl_gemm_schedule_and_locality.md` · `flydsl_gemm_epilogue.md` · `flydsl_gemm_correctness.md` ·
`flydsl_gemm_attribution.md` · `../../../../hardware/mi350_lds.md` ·
`../../../../common_methodology/optimization/lever_lds_banks.md`
