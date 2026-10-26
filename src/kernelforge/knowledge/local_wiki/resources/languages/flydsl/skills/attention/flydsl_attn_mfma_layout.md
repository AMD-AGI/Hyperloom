---
title: FlyDSL attention — MFMA timing, atom choice, fragment layouts and fp8 MFMAs on gfx950
kind: language
lever: flydsl_attn_mfma_layout
gens: [gfx950]
dtypes: [bf16, fp8]
regimes: [prefill, decode, backward, causal]
updated: 2026-10-26
sources:
  - ROCm/FlyDSL@dd8373349:kernels/attention/flash_attn_utils.py:3500
  - ROCm/FlyDSL@dd8373349:kernels/attention/flash_attn_utils.py:3732
  - ROCm/FlyDSL@dd8373349:kernels/attention/flash_attn_utils.py:182
  - ROCm/FlyDSL@dd8373349:kernels/attention/flash_attn_utils.py:3199
  - ROCm/FlyDSL@dd8373349:kernels/attention/flash_attn_utils.py:5560
  - ROCm/FlyDSL@dd8373349:kernels/attention/flash_attn_gfx950.py:520
  - ROCm/FlyDSL@dd8373349:python/flydsl/expr/rocdl/cdna4.py:80
  - ROCm/FlyDSL@dd8373349:lib/Dialect/FlyROCDL/CDNA4/MmaAtom.cpp:245
---

# MFMA atoms and fragment layouts in a FlyDSL attention kernel

> Attention chains GEMMs through registers: S (or dP) leaves one MFMA as a C fragment and enters the next one
> as an operand. The atom therefore fixes the softmax lane map, the reductions, the store and the register
> footprint, not only the MAC rate. GEMM-side atom facts are in `../gemm/flydsl_gemm_mainloop.md` §5, per-lane
> register counts per atom in `../../../../hardware/mi350_matrix_core.md`, transposed-read lane maps in
> `flydsl_attn_lds.md` §3.
>
> Numbers marked *(field)* are measured values. Use them for direction and order of magnitude, not as targets.

## Route here when
- You are choosing the atom of an attention GEMM, or porting a kernel from one atom to another.
- You need the lane ↔ (row, column) map of an operand or accumulator to write a mask, a reduction, a pack or
  a store.
- You are counting how much softmax work fits under the MFMAs.
- You are considering fp8 for one of the GEMMs.

## 1. MFMA timing on gfx950 (measured)
| Instruction | Pipe cycles | Issue blocked | VALU shadow per MFMA |
|---|---|---|---|
| `v_mfma_f32_16x16x32_bf16` | 16 | 8 | 8: one `v_exp_f32` or two `v_mul_f32` |
| `v_mfma_f32_32x32x16_bf16` | 32 | 8 | 24: three exps or six multiplies |
| `v_mfma_f32_16x16x16_bf16` | 16 | 4 | 12: one exp or three multiplies |

- An MFMA followed by k VALU ops takes `max(pipe cycles, X + 8·k_exp + 4·k_mul)` cycles in one wave, with
  X = 8 for the two 128-bit-operand atoms and 4 for 16x16x16; every measured point fit exactly. Each exp or
  multiply beyond the shadow adds its full cost.
- A chain dependent through srcC (srcC = previous vdst) runs at the pipe rate: the compiler inserts no `s_nop`,
  and the chain costs nothing extra. Two or four independent accumulators run at the same 16 or 32 cycles per
  instruction. Splitting an accumulation into independent chains buys no latency ILP.
- One MFMA pipe per SIMD: MFMA streams of two waves on one SIMD run one after the other, and the shadow is
  shared by every wave on that SIMD. exp and mul draw on the same shadow (`flydsl_attn_softmax.md` §1).
- `16x16x16_bf16` takes as long as `16x16x32` for half the K, i.e. half the MAC rate. The "4-cycle issue,
  12-cycle shadow" figure belongs to it, not to 16x16x32.
- 16x16x32 and 32x32x16 have the same MAC rate (512 MAC per cycle), but per MAC the 32x32x16 atom leaves 1.5×
  the VALU shadow (24 of 32 cycles against 8 of 16).
- **Softmax budget.** A 32 kv × 32 q tile at head dim D needs D/8 `32x32x16` MFMAs for QK and P·V together,
  i.e. 3·D shadow cycles, against 16 `v_exp_f32` per wave (128 cycles) plus the max, shift, pack and sum. At
  D = 64 the exps alone take two thirds of the shadow; at D = 128 one third. FlyDSL's forward groups one
  `32x32x16` with three exps (`_sched_barrier_exp_pairs(traits, 6, 3, …)`), exactly one shadow.
- These are pipe facts. At the power cap rank atoms and schedules by wall time and sclk
  (`flydsl_attn_attribution.md` §8), and use the shadow table only to prefilter
  (`flydsl_attn_registers_occupancy.md` §8).

## 2. An MFMA ignores EXEC
- On gfx950 an MFMA computes and writes all 64 lanes whatever EXEC holds. With EXEC = 0, one lane or half the
  lanes, `v_mfma_f32_16x16x32_bf16` still wrote its full result to every lane, and the operands of masked-off
  lanes contributed to it; a `v_mov_b32` in the same region honoured EXEC.
- A lane mask or a divergent `if` neither skips an MFMA nor protects its destination. Skip MFMA work with a
  wave-uniform branch, and mask scores with `-inf` (exp2 then gives exactly 0). Per-wave classification of
  diagonal blocks: `flydsl_attn_backward.md` §8.

## 3. Fragment layouts (bf16, wave64)
`D[m][n] = Σ_k A[m][k]·B[k][n] + C[m][n]`; s indexes the 8 bf16 values of an operand register group, t and e
the f32 accumulator elements of a lane.

| Atom | A | B | C / D |
|---|---|---|---|
| 16x16x32 | m = lane%16, k = (lane//16)·8 + s | n = lane%16, k = (lane//16)·8 + s | n = lane%16, m = (lane//16)·4 + t, t < 4 |
| 32x32x16 | m = lane%32, k = (lane//32)·8 + s | n = lane%32, k = (lane//32)·8 + s | n = lane%32, m = (e//4)·8 + (lane//32)·4 + e%4, e < 16 |

- In C the column n is the lane and the row m runs over the lane's elements: a lane owns part of one column.
  16x16x32 gives a lane 4 consecutive m; 32x32x16 gives it four runs of 4 consecutive m, the neighbouring runs
  sitting in lane ^ 32.
- Prove a map element by element with a position-encoded input (a ramp in one operand, ones in the other).
  Keep the ramp exactly representable in bf16 (integers ≤ 256). Row-sum and sum-over-k checks are
  permutation-invariant. *(field)* A 32x32x16 port passed `rowsum(P) = 1`
  and probes that varied C along one index only, while its operand orientation and pack order were wrong.

## 4. Orientation: compute Sᵀ and keep P inside the lane
- **Sᵀ.** Issue QK with A = K and B = Q. C is then Sᵀ: n = q is the lane, m = kv runs over the elements, so a
  lane holds the scores of one q row (16 kv per 32x32x16 tile, the other 16 in lane ^ 32). Row max and row sum
  are in-lane folds plus one `permlane32_swap`, and per-row values (running max, l, LSE, delta) are one per
  lane, with no gather. FlyDSL's gfx950 forward computes QK this way (`qk`) and reduces with one
  `permlane32_swap` (`_reduction_pair`).
- Both QK operands stay contiguous in d (`[kv, d]` and `[q, d]` in LDS); the only transposed read is Vᵀ for
  P·V (`flydsl_attn_lds.md`).
- **P or dS feeding the next GEMM is locked to the previous GEMM's C layout.** Orient the previous GEMM so the
  next GEMM's contraction index is C's element index m, never the lane index n. P then packs inside the lane:
  FlyDSL's forward passes P as B of P·V with A = Vᵀ. With the contraction index on the lane, every tile needs a
  cross-lane transpose.
- The k order is free as long as both operands of an MFMA agree. Address the other operand's transposed reads
  (two `ds_read_b64_tr_b16` per 8-wide fragment) so they produce C's kv order, `(e//4)·8 + (lane//32)·4 + e%4`
  for 32x32x16; the 16 C elements then pack straight into two B registers. *(field)* A pack that added a
  `permlane32_swap` on top of matching orders exchanged kv 4–7 with 8–11 and halved dQ. For 16x16x32, two
  adjacent 16-kv C tiles give a lane the 8 kv of one B register.
- **Store.** A 32x32x16 O fragment holds runs of 4 consecutive d per lane; FlyDSL's forward swaps halves with
  `permlane32_swap` to assemble 8 consecutive bf16 (16 B) per lane per store (`_store_permlane_o`). A
  16x16x32 fragment's 4 consecutive values go out directly as one 8-byte store.

## 5. Choosing the atom per GEMM
- Dump the atom of every GEMM in both passes before reasoning: the forward and the backward of one code base
  can use different atoms. Changing the atom is a port — fragment layout, transposed-read addressing,
  register footprint and softmax lane map all change. Read the ISA delta (`v_mfma` count, VGPR, spill,
  `ds_read` mix) before timing.
- **Backward: 16x16x32.** *(field)* Rewriting a dK/dV kernel from 32x32x16 cut its time 1415 → 1168 µs and the whole
  backward gained 8.7% (`SQ_WAIT_ANY` −47%; a full rewrite, the cause was not isolated); a dq body ran 1204 TF/s
  against 1088–1093, and even without exps the correct 32x32x16 version (1198) stayed below 16x16x32 (1220); one GEMM
  switched back to 32x32x16 alone cost 12%. What does hold either way:
  - the C fragment is 4 f32 per lane instead of 16, so the S, dP and dS of a tile live in small pieces and the
    transient register peak is lower *(field: a 32x32x16 dq port spilled until restructured to one kv tile at a
    time)*;
  - its 4 consecutive values per lane store without a permlane;
  - the accumulator registers for a given output block are the same for both atoms.
  Independent-chain latency ILP is not a reason (§1).
- **Forward: 32x32x16.** FlyDSL's gfx950 forward uses `fx.rocdl.MFMA(32, 32, 16, …)` for QK and P·V. It
  leaves 1.5× the VALU shadow per MAC (§1), needs one swap per row reduction (§4) and keeps one q row per lane.
  *(field)* A D=64 forward ported to 16x16x32 ran slower, by up to ~8%.
- The GEMM finding that a smaller atom moves fewer register-file bytes per MAC and can clock higher at the power
  cap (`../gemm/flydsl_gemm_mainloop.md` §5) applies here too: rank atoms on wall time, sclk and power, never on
  MFMA utilization.

## 6. Row sums on the matrix pipe
- The softmax row sum can be an MFMA with a constant all-ones A over the packed P. Use 16x16x32: it needs half
  the pipe cycles of a 32x32x16 row sum; by the measured kernel's own accounting a 32x32x16 row sum would cost
  more than the VALU adds it replaces.
- Layout, with packed P as B of a 32x32x16 P·V (n = q = lane%32, k = kv): read the same registers as B of a
  16x16x32. Lane groups 0 and 2 then hold the two kv halves of q = n, groups 1 and 3 those of q = n + 16. Fill
  A with bf16 1.0 (`0x3F803F80` per dword) in lanes where `lane%4 == 0 and (lane//4)%2 == (lane//16)%2`, zero
  elsewhere. Element 0 of each lane's D is the sum for its own q = lane%32, half-wave partner included: the adds
  and the `permlane32_swap` disappear.
- *(field)* Per iteration `v_add_f32` 68 → 0 and `permlane32_swap` 2 → 0 for 8 new 16x16x32 MFMAs; VGPR
  162 → 150; +1.96%; SNR 51.26 → 51.61 dB because numerator and denominator share the bf16 P.
- These MFMAs anchor register allocation: removing or replacing them raised spills sharply
  (`flydsl_attn_registers_occupancy.md` §4). *(field)* Restoring an MFMA row sum in a different online forward
  was within noise; price it per kernel.
- An MFMA that reads P straight from an inline-asm pack needs padding the compiler cannot add
  (`flydsl_attn_schedule_sync.md`). *(field)* Symptom: non-deterministic results with errors on half of the q
  rows.

## 7. More rows per wave
- Two M tiles per wave sharing every K/V fragment halve the operand reads per MAC but double the output state:
  O is `ROWS·D/64` and Q `ROWS·D/128` dwords per lane. Two q heads or two row groups cost the same registers.
- *(field)* D=64 forward, rows per wave doubled: non-MFMA instructions per unit of work 194 → 151.5, sclk +9.2%
  at the power cap, but loop-carried VGPRs 100 → 192, occupancy 4 → 2 waves per SIMD, cycles +11.5%: net
  −0.86%. Price the occupancy step first (`flydsl_attn_registers_occupancy.md` §3).

## 8. fp8 MFMAs
- Only the wide-K f8f6f4 forms raise throughput: 16x16x128 and 32x32x64 do 4× the K of the bf16 atoms in 2×
  the cycles. The legacy fp8 forms (`16x16x32_fp8_fp8`, `32x32x16_fp8_fp8`) take the bf16 cycle counts for the
  same K *(field: 32 cycles per `32x32x16_fp8_fp8`, the same as bf16)*. Name the shape whenever an fp8 speed-up
  is quoted. *(field)* Moving a forward's P·V to the legacy fp8 form bought no MFMA throughput.
- FlyDSL v0.3.2 reaches the wide-K forms only in scaled form: `rocdl.mfma_scale_f32_16x16x128_f8f6f4`,
  `rocdl.mfma_scale_f32_32x32x64_f8f6f4`, or the atom `rocdl.cdna4.MFMA_Scale(16, 16, 128, …)`, with E8M0
  scales (bias 127). The unscaled wide-K intrinsics have no ROCDL binding.
- A GEMM qualifies only with a contraction ≥ 128. At D = 64, QK and dP (contraction D) and dV and dK
  (contraction BLOCK_Q = 64) do not; dQ (contraction BLOCK_KV) can. *(field)* Storing only the dQ partials in
  fp8 saved bytes, not MFMA time: +4.3% with dq SNR 47.9 → 31.3 dB.
- A scale must be constant along the contraction axis to leave the MFMA. A tensor contracted along different
  axes in two GEMMs — dO along d in dP = dO·Vᵀ and along q in dV = Pᵀ·dO; dS along q and along kv — needs a
  per-tensor scale or two quantized copies. That is why fp8 attention defaults to per-tensor scales.
- fp8 operands do not shrink the f32 accumulators; the S/dP register peak stays where it was.
- Treat accuracy as a budget: error powers of independent quantizations add. *(field)* Q and K dominate (an S
  error is amplified by the exponential), P and dS are the cheapest; finer scales recover only ~0.8–0.9 dB
  because the e4m3 mantissa dominates; e5m2 is ~6 dB worse; fp8 P at D = 64 is estimated near 27.6 dB, below a
  34 dB gate.
- fp8 P also constrains the softmax shift (`flydsl_attn_softmax.md` §6.1, §6.5).

## Verify
- The atom of every GEMM in both passes is read from the ISA, not assumed.
- Lane maps and k orders are proven element by element with position-encoded inputs.
- No MFMA relies on EXEC to be masked or skipped.
- An atom or rows-per-wave change reports VGPR, spill, occupancy and sclk next to the wall time.

## Related
`flydsl_attn_softmax.md` · `flydsl_attn_lds.md` · `flydsl_attn_registers_occupancy.md` ·
`flydsl_attn_schedule_sync.md` · `flydsl_attn_backward.md` · `flydsl_attn_attribution.md` ·
`../gemm/flydsl_gemm_mainloop.md` · `../../../../hardware/mi350_matrix_core.md`
