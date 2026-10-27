---
title: FlyDSL grouped GEMM — fp8 tensorwise, MXFP8 and MXFP4
kind: language
lever: flydsl_grouped_lowprec
gens: [gfx950]
dtypes: [fp8, mxfp8, mxfp4]
regimes: [forward, dgrad, wgrad, skew, fused-epilogue, quantization]
updated: 2026-10-27
sources:
  - AMD-AGI/Primus-Turbo@ca1fdd67:primus_turbo/flydsl/grouped_gemm/grouped_gemm_fp8_kernel.py
  - AMD-AGI/Primus-Turbo@ca1fdd67:primus_turbo/flydsl/grouped_gemm/grouped_gemm_mxfp8_kernel.py
  - AMD-AGI/Primus-Turbo@ca1fdd67:primus_turbo/flydsl/grouped_gemm/grouped_gemm_mxfp4_kernel.py
  - AMD-AGI/Primus-Turbo@ca1fdd67:primus_turbo/flydsl/grouped_gemm/grouped_gemm_fp8_glu_kernel.py
  - AMD-AGI/Primus-Turbo@ca1fdd67:primus_turbo/flydsl/grouped_gemm/grouped_gemm_mxfp8_glu_kernel.py
  - AMD-AGI/Primus-Turbo@ca1fdd67:primus_turbo/flydsl/grouped_gemm/grouped_gemm_mxfp4_glu_kernel.py
  - AMD-AGI/Primus-Turbo@ca1fdd67:primus_turbo/pytorch/ops/grouped_gemm_fp8.py
---

# Low precision in a FlyDSL grouped GEMM

> What fp8 with per-tensor scales, MXFP8 and MXFP4 add to a grouped GEMM: per-group scale tensors and their
> preshuffle, scale packing, scale delivery, quantization cost, padding, the C store, and GLU/dGLU fusion. The
> scale and MFMA mechanics that are the same as in a dense GEMM are in `../gemm/flydsl_gemm_datapath.md` §11 and
> `../gemm/flydsl_gemm_mainloop.md` §5–§6 and are not repeated. "FP8", "MX8", "MX4", "FP8G", "MX8G", "MX4G" are
> Primus-Turbo's `grouped_gemm_{fp8,mxfp8,mxfp4}{,_glu}_kernel.py` under `primus_turbo/flydsl/grouped_gemm/` and
> "OPS8" its `pytorch/ops/grouped_gemm_fp8.py`, at main `ca1fdd67`.
> Numbers marked *(field)* are measured values. Use them for direction and order of magnitude, not as targets.

## Route here when
- You add fp8 or MX formats to a grouped kernel, or a quantized grouped kernel is slower than its bf16 or
  tensorwise sibling.
- A scale preshuffle, quantizer or padding kernel shows up next to the GEMM in a trace.
- You are choosing the C store or cache policy for grouped outputs, or fusing a gated activation.

## 1. What the group dimension adds
- **Tensorwise fp8**: two scalars per GEMM, applied once in the epilogue (`a_scale * b_scale`); nothing per group.
- **MX**: E8M0 scales per 32 contraction elements and per group. The quantizer pads each group to the scale
  granule, so the kernel reads through a padded offset table and writes through a tight one: MX8 takes
  `group_offs` (padded read) and `group_offs_out` (tight write, MX8:1244-1247). Its preshuffle stages each group
  into its own 64-row slabs so a 32-aligned group base needs no 64-alignment from the quantizer (MX8:263-264).
- **MX dgrad runs as NT on a column-quantized weight** (OPS8:806): the preshuffled block scale of B is "only
  walkable by the NT addressing" (MX4G:21-23). Tensorwise dgrad stays NN with transpose reads.
- **MX wgrad takes column-quantized operands `[OUT, M_total]`**; K tails then need a run-time guard
  (`flydsl_grouped_wgrad.md` §3).
- **Quantization runs every call for activations and gradients.** Weights quantize once per optimizer step and
  the backward reuses them; a benchmark that passes high-precision weights re-quantizes them per call. *(field)*
  Pre-quantizing weights took one fused MoE MLP forward from 4.28 to 3.47 ms; the backward did not change.

## 2. The scale preshuffle is a per-call helper kernel
Scales arrive per call (activations, gradients), so the repack into the MFMA layout runs every call. Launch it
from the same `@flyc.jit` stub as the GEMM so it costs no Python dispatch (MX8:1101-1120;
`../gemm/flydsl_gemm_datapath.md` §11). Then find out what bounds it.
- **E8M0 repack (MXFP8) is dispatch- and latency-bound.** *(field)* On the measured build it took 0.8–1.5% of the
  call's wall time at an effective 2.6–2.9 TB/s, a third of HBM bandwidth. Only cutting the workgroup count
  paid: one workgroup taking two 64-row slabs halved the A-side workgroups, +1.04% on the scored total. More
  threads per workgroup (256 → 512) was 24–28% slower (twice the waves meet at each barrier); dword → dwordx4
  reads were net zero. The binding budget is LDS per resident workgroup (160 KB / 8 = 20 KB at eight per CU), not
  the 160 KB total: pair slabs only while eight still fit. The staged row pitch moves the best k-tile, so
  re-sweep it after a pitch change.
- **MXFP4 repack can be VALU-bound.** *(field)* A per-thread O(G) group scan to map 64-row slabs back to source
  rows ran 759 VALU per wave for 8 useful dwords per thread; the lane-resident group table of
  `flydsl_grouped_group_mapping.md` §5 cut it to ~206 and the kernel from 35.3 to 16.3 µs, +1.1% geomean over 18
  cells. Rule: more than ~10 VALU per useful dword means index arithmetic owns the kernel; widening stores or
  staging in LDS will not help.
- **Its cost is not what removing it saves.** *(field)* Turning the MXFP8 repack off made the call 6 µs slower: it
  had warmed the scale workspace in L2 for the GEMM. A trace overstates such kernels 2–3× against their
  in-stream cost (`flydsl_grouped_host_autotune.md` §7).

## 3. Pack four E8M0 per dword — per group on wgrad
- `op_sel` picks one of four bytes, so four is the hardware ceiling (`../gemm/flydsl_gemm_datapath.md` §11).
- **NT (forward, dgrad).** The k-step's byte is `k % 4`, known at compile time. MX8 packs NT only
  (`_GG_SCALE_PACK = 4`, MX8:74-76). *(field)* The repack took 32% less time (14.3 → 9.7 µs), the GEMM was
  unchanged and bit-exact, end to end ≈1%: on that build the repack was ~3.5% of the forward/dgrad time (§2
  has another build's share).
- **Variable-K wgrad.** A group's contraction starts at any 128-element boundary, so a global `k % 4` byte select
  is wrong for most groups; Turbo's MXFP8 wgrad stays unpacked for that reason. Pack each group from its own
  start instead: base `ks0 // 4 + g` for a group starting at 128-block `ks0`, row-group stride `K128 // 4 + G` (the
  `+g` / `+G` is the at most one partial dword per group; `floor(a/p) + ceil(n/p) ≤ floor((a+n)/p) + 1` keeps the
  regions disjoint). The alignment work moves into the repack kernel, which runs once per call; the GEMM adds one
  integer add.
  - *(field)* MXFP8 variable-K wgrad, six cells: +4.7–9.3% (scale loads were exposed there; VMEM per workgroup
    −14.2%, the byte selects went to the separate scalar port).
  - *(field)* Packing that assumed 512-aligned group starts read 4.09 and −2.44 dB SNR on unaligned groups while
    the determinism gate and a check that the repacked and raw-scale paths agree both passed. The per-group form read
    55.56–55.64 dB on 16 layouts (128- but not 512-aligned starts, empty first/middle/last groups, ragged G=32),
    and pack 1 and pack 4 gave byte-identical outputs. Gate packing with unaligned group starts.
- **Where packing buys nothing.** *(field)* In a dense per-K loop whose scales were already prefetched one step
  ahead, a probe that reused the k=0 scales measured the scale tax at ≈0. Pack where the scale loads are exposed.

## 4. Scale delivery in NT: the floor for its scale-slab layout
- *(field)* MXFP8 8-wave NT: scale loads to VGPRs and the operand DMA share `vmcnt`, and each K iteration's drain
  retires both, so a scale prefetch can reach at most one iteration ahead; prefetching a whole pack group, or
  spreading the scale loads evenly, measured +0.17% and +0.08%, inside the noise. The main loop's scale loads were
  already dwordx4; the narrow loads in the ISA histogram were the prologue's group-offset scan. Split ISA counts
  by region first.
- *(field)* MXFP4 whole loop: removing the two scale `buffer_load_dwordx4` per phase was worth 4.69%; a probe that
  froze the scale addresses (same instructions) split it into 1.79% memory and 2.90% structure. Deeper scale
  prefetch was flat. One scale dword serves four MFMAs through `op_sel`; eight row blocks × two sub-steps need four
  dwords per operand per phase, and dwordx4 is the widest buffer load: two loads per phase is the floor for this
  slab layout. Going lower means re-laying the slab (two dwords per phase, each serving two MFMAs), which moves
  bytes into the preshuffle.
- The scaled MFMA itself costs nothing extra. *(field)* Dense: scaled 3068 vs unscaled 3093 TF/s; grouped wgrad
  807 → 807 µs with the scale operand removed. Scale cost is load issue and registers, not the instruction.
- MX tile heights are fixed by the scale layout. *(field)* In an MXFP4 grouped kernel whose A-scale repack groups
  64 rows and whose wave regions walk 128, `BLOCK_M` must be a multiple of 128: a 288-row tile puts a wave at row
  144 and selects the wrong byte. Re-lay the scale slab before trying odd tile heights.

## 5. Quantization: a large share of the forward, a small lever
- *(field)* Grouped MX quantization was 35–42% of the forward's absolute time on four shapes at 4096 tokens per
  group. A bit-exact fused dual-cast quantizer still left the backward unchanged and moved the forward little: 60%
  of the forward's gap and 90% of the backward's gap sat in the GEMMs. A share of time is not a share of the gap,
  and the difference between two speed ratios is neither.
- **Fuse the per-call metadata into one prologue kernel.** Each quantizer workgroup ran an O(G) group search over
  three offset tables; in a short memory-bound kernel that exposed ~42 µs. One prologue thread per M block does
  the search once and writes the block's row base, offset and end, launched back to back in the same `@flyc.jit`
  stub. *(field)* Kernel 155 µs, call 181 µs, faster than the HIP quantizer. Moving the search to host-side torch
  ops gave a 153 µs kernel and a 236 µs call: ~10 tiny launches added 80–120 µs.
- **Quantize all expert weights in one launch.** Batch the dense quantizer over `[G, N, K]` with a per-batch scale
  byte base, dword-aligned so neighbouring experts' packed scales never share a dword. *(field)* 1.3–1.54× faster
  than the HIP kernel; a Python loop over experts was 2.7–4.3× slower than HIP.

## 6. Padding goes on the contraction dimension
- A row pitch ≡ 64 mod 128 B makes every other row straddle two lines (`../gemm/flydsl_gemm_datapath.md` §6, §10).
  *(field)* Tensorwise fp8 grouped forward, G=32 balanced, K and N both ≡ 64 mod 128: padding neither / K
  only / N only / both gave 2196 / 2554 / 2202 / 2554 TF/s. The whole +16% came from the contraction pad; the N
  pad matters for dgrad, where N becomes the contraction. Count FLOPs with the real dims.
- *(field)* The same pad (to 128) for a fused SwiGLU MLP whose hidden and expert widths are both ≡ 64 mod 128:
  forward 8.2% and backward 2.6% less time.
- **When the operand cannot be padded:** FP8's NN dgrad reads each row's enclosing line-aligned window when
  `K % 128 == 64` (`nn_kshear`, FP8:218, 277-280; its comment reports +25.7% L1→L2 read requests without it).
  For MX, pad only the scales: zero E8M0 past K makes over-read terms exactly zero, so the operand keeps its real
  K (`k_real`, MX4:364-373; `../gemm/flydsl_gemm_mainloop.md` §6).
- Never re-pad a weight per call: `flydsl_grouped_host_autotune.md` §4. Free-dim padding for wgrad outputs:
  `flydsl_grouped_wgrad.md` §12.

## 7. The C store for grouped outputs
The store classes and their measured prices are in `../gemm/flydsl_gemm_epilogue.md` §2–§4. What grouped adds:
- **The store is bytes, not VALU.** *(field)* MXFP4 grouped NT: deleting the C-store epilogue saved 61% of the
  per-tile fixed cost (4.87 of ~7.9 µs per tile, fitted on that build; the F/P fit is in
  `flydsl_grouped_tile_pricing.md` §6); deleting 254 `v_accvgpr_read` and 254 conversions while keeping every
  store saved 0.5%, inside the noise.
- **With narrow (2-byte) folded stores, do not issue consecutive stores to the same 128 B line.** *(field)*
  MXFP4 grouped at one wave per SIMD: grouping folded stores by output row so four in a row fill one line:
  −0.76% and −0.52%; issuing by accumulator, four lines in turn, was faster.
- **Cache-policy bits on C (grouped MX kernels, beta=0 stores): only `nt`, only on whole-row stores, and by the
  number of slabs competing for L2.**
  - *(field)* 2-byte scalar stores with `sc0` (aux = 1): 8 of 8 MXFP8 NT cells 0.2–1.7% slower.
  - *(field)* Narrow folded 2-byte stores with `nt`: −4.5%; DRAM write requests +54%, because a store's 32 B pieces
    merge into 64 B writes only in L2.
  - *(field)* Whole-row folded stores (16 lanes × 8 B fill a 128 B line) with `nt` at G=32 thin groups: wgrad
    +2.1–2.4%, NT +0.88 points; L2 hit 59.2 → 61.2%, DRAM reads −72 to −99 MB, write bytes unchanged. At G=4
    with large M the same bit was flat or slightly negative. The gain is the weight residency C stops evicting,
    so it scales with the expert slabs live in L2 at once; choose it from the runtime shape. MX4 keeps a
    separate thin-group table entry with `nt` on (`_GMXFP4_NT_CFG_THIN`, MX4:1100-1101; selector
    MX4:1164-1170).
  - *(field)* `sc1` alone or added to `nt` cost 4–5% on six wgrad cells: it sends the writes past the per-XCD L2
    that merges them.
- E8M0 scale stores are never non-temporal (`../gemm/flydsl_gemm_epilogue.md` §5).

## 8. Comparing MX with tensorwise
- **A fair comparison.** Both on the same GEMM backend; the tensorwise path in its native layouts (an option that
  forces NT in the backward adds transposes of both operands); quantization inside the timed region, fresh each
  step. MX's reference is a scaled MX GEMM, not an unscaled per-tensor one.
- **Find what the gap is made of before attacking it.** Compare memory instructions per MFMA, steady-state time per
  K iteration, synchronization instructions per MFMA and K iterations per tile. *(field)* An MXFP8 8-wave NT kernel
  against the tensorwise 4-wave wgrad it had to beat led on the first two and lost on the last two
  (`flydsl_grouped_tile_pricing.md` §2, §6); DRAM bytes, L1→L2 requests and power were level. Its extra
  synchronization is not free to delete: the 8-wave barriers keep the waves in phase (§2 there).
- **Instruction trimming in the steady state pays only when non-MFMA issue demand approaches the MFMA shadow.**
  *(field)* Cutting the MXFP4 grouped loop from 442 to 419 (NT) and 410 to 387 (wgrad) instructions, all SALU,
  changed wall time by 0: non-MFMA issue demand was 0.25–0.28 of the MFMA shadow. Below ~0.5, do not invest.
- At the power cap a cycle saving converts to wall at a shape-dependent rate
  (`../gemm/flydsl_gemm_attribution.md` §4).

## 9. Why MXFP4 grouped does not reach 2× MXFP8
- *(field)* The fp4/fp8 grouped speed ratio was 1.32–1.69 (geomean 1.51 over 18 cells): lowest on shallow-K wide
  shapes, closest to 2 on deep K. The fp4 kernel ran 4-wave workgroups at one wave per SIMD (the fp8 kernel 8-wave
  workgroups at two) and its MFMA pipe starved for operands: MfmaUtil was 36.6% on the lowest-ratio cell and
  51.6% on the highest. Compare utilization counters across wave counts only after
  `../gemm/flydsl_gemm_attribution.md` §9.
- Raising fp4 occupancy by splitting the 256×256 tile over eight waves lost with both pipeline forms
  (`flydsl_grouped_tile_pricing.md` §2). Raise the per-wave feed at one wave per SIMD instead.

## 10. GLU and dGLU fused into the epilogue
- **Accept on fused total against unfused total**, never on the fused kernel's TF/s against a bare GEMM
  (`../gemm/flydsl_gemm_epilogue.md` §8). *(field)* Fused glu (GEMM + SwiGLU) ran ~14–15% and fused dglu ~28–29%
  slower than a bare GEMM, and both were net wins on the total. The gap is structural: glu writes a third stream
  (the activation) that one workgroup per CU cannot hide; dglu must stream the pre-activation in and its gradient
  out, about twice the HBM traffic (a 24% roofline floor at 8 TB/s). `waves_per_eu` 2/3/4 changed nothing on glu.
- **Gated activations with a padded half: compute at the pitch, store at the real width.** With gate and up halves
  of width `I` padded to `Ip`, the weight shape, loop bounds and the up half's column offset use `Ip`; the store
  width, column masks and the compact `[M, 2I]` output use `I`, and pad columns are written as zero. Plain zero
  propagation is right for forward and dgrad but leaves the weight gradient with pad rows between gate and up.
- **Port the plain kernel's store wins into the fused epilogue.** *(field)* Paired-column stores in fused glu:
  +3.7%; it has three output streams to gain on.
- **Measure the kernel you changed, not an average of glu and dglu.** *(field)* A +0.55% glu-only gain, clean over
  five readings, was invisible in the averaged score.
- **Turbo's forms.** fp8: the forward emits the pre-activation and the activation together, and the fc2 dgrad
  consumes the activation gradient in registers so it never reaches HBM, both through the NT/NN mainloops behind
  `glu`/`dglu` flags (FP8G:14-21). MXFP8: GEMM, then a separate GLU + dual-quantization kernel, because folding
  the epilogue "pushes the tile past 128 VGPRs and into scratch" (MX8G:59-61). MXFP4: both directions NT
  (MX4G:21-23). Operand pairing for gate/up in one lane: `../gemm/flydsl_gemm_epilogue.md` §8.

## Verify
- Packed-scale kernels are gated with unaligned group starts and random E8M0 exponents, against a reference.
- Each helper kernel (repack, quantizer, padding) was priced in the stream and launches from the GEMM's stub.
- The padded dimension of every operand is its contraction dimension, checked per pass.
- `nt` on C is enabled only for whole-row stores and was measured at the deployed group count.
- Fused GLU/dGLU was accepted on total time, with pre-quantized weights.

## Related
`flydsl_grouped_wgrad.md` · `flydsl_grouped_group_mapping.md` · `flydsl_grouped_tile_pricing.md` ·
`flydsl_grouped_host_autotune.md` · `flydsl_grouped_correctness.md` · `flydsl_grouped_measurement.md` ·
`../gemm/flydsl_gemm_datapath.md` · `../gemm/flydsl_gemm_epilogue.md` · `../gemm/flydsl_gemm_mainloop.md` ·
`../gemm/flydsl_gemm_attribution.md`
