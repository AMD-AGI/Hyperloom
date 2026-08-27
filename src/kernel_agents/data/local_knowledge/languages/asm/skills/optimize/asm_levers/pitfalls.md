---
title: ASM / MFMA pitfalls & anti-patterns
kind: language
gens: [gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz, fp8_e4m3]
regimes: [both]
status: sota
updated: 2026-06-08
sources:
  - https://github.com/ROCm/HIP/issues/3333
  - https://github.com/llvm/llvm-project/issues/131954
  - https://github.com/iree-org/iree/issues/23765
  - https://rocm.blogs.amd.com/software-tools-optimization/measuring-max-achievable-flops-part2/README.html
---

# ASM / MFMA pitfalls

## TL;DR
The low level bites in predictable ways: MFMA in asm defeats the SW pipeliner; inline-asm clobber bugs;
32×32 MFMA clocking lower than 16×16; LDS bank conflicts; spurious accumulator spills; fp8 encoding
mismatch; and `s_waitcnt` off-by-ones. Most are diagnosable from the disassembly.

## The pitfalls
1. **MFMA in inline asm → no SW pipelining.** `SchedGroupMask` only sees *intrinsic* MFMA. Hand-writing
   `v_mfma` in `asm volatile` blinds the pipeliner. Keep MFMA as the intrinsic; hand-schedule only the
   surrounding `buffer_load`/`ds_read`.
2. **Inline-asm register clobber** (HIP #3333). Multiple `volatile` blocks collapse into the same regs /
   get reordered. Rules:
   - **One asm block** for ordered sequences.
   - **Early-clobber `"=&v"`** when an output reg must not alias an input (the classic "first load
     clobbers `v[0:1]`, later loads break" bug).
   - **`"memory"` clobber + `volatile`** around timing/sync code, or `-O2`+ reorders/deletes it (e.g.
     `s_memtime` latency probe gives wrong results without it).
   ```cpp
   asm volatile(
     "global_load_dwordx4 %0, %2, off\n"
     "global_load_dwordx4 %1, %3, off\n"
     "s_waitcnt vmcnt(0)\n"
     : "=&v"(v0), "=&v"(v1) : "v"(ptr0), "v"(ptr1) : "memory");
   ```
3. **Defaulting to 32×32 MFMA.** `mfma_16x16x16` usually yields higher *achievable* FLOPs on MI300X — the
   32×32 op draws more power → lower clock (ROCm Max-Achievable-FLOPs Part 2). Default 16×16; test 32×32.
4. **LDS bank conflicts.** 32 banks × 4B; the `ds_read` feeding MFMA must avoid lane→bank collisions.
   Fix with **XOR swizzle** of the LDS *write* address (`make_xor_transform`, no extra LDS — preferred)
   rather than **LDS padding** (costs extra LDS → lower occupancy; AMD tuning guide warns of this).
5. **Spurious accumulator spills** (LLVM #131954). At large tiles the compiler inserts unnecessary
   `v_accvgpr_read/write` and/or `scratch_` spills → TFLOP/s plateaus or regresses as the tile grows.
   Grep `.s` for `accvgpr`/`scratch_`; shrink tile / let acc stay in VGPR. See
   [register_alloc.md](register_alloc.md).
6. **`s_waitcnt N` off-by-one.** `vmcnt(N)`/`lgkmcnt(N)` = "wait until **≤N** remaining", not "wait N
   instructions". A wrong N is a data race (too low) or a stall (too high).
7. **fp8 encoding mismatch.** CDNA3 = **fnuz** fp8 (different bias); CDNA4 = OCP fp8/MXFP. Mismatched
   dequant scale → silent garbage.
8. **Direct-to-LDS assumptions.** DGL `buffer_load ... lds` + the scaled-GEMM DGL path is primarily a
   **gfx950** story; on gfx942 the two-step `buffer_load`→`ds_write` staging is still common. Don't assume
   DGL is free on MI300X (iree #23765 — DGL/XOR-swizzle/pad tradeoffs).
9. **Wrong MFMA fragment placement.** Lanes pack A/B/C with no guaranteed element order. Guessing the
   layout → silent wrong answer. Use the calculator.
10. **Wave specialization** (producer/consumer waves) **underperforms on CDNA3/4** — AMD's static register
    allocation means producer waves hold registers without computing (HipKittens: ~80% of peak BF16 GEMM
    on MI355X). Use ping-pong / interleave scheduling instead.

## Verify
```bash
amdclang++ -x hip --offload-arch=gfx942 -O3 -S kern.cpp -o kern.s
grep -E 'v_mfma|s_waitcnt|accvgpr|ds_read|buffer_load|scratch_' kern.s
# rocprof LDSBankConflict counter for #4; -Rpass-analysis=kernel-resource-usage for #5
```

## Sources
- HIP #3333 (inline GCN asm multi-load register clobber pitfalls): https://github.com/ROCm/HIP/issues/3333
- LLVM #131954 (large MFMA tiles → spurious v_accvgpr / spills): https://github.com/llvm/llvm-project/issues/131954
- iree #23765 (direct-to-LDS + XOR-swizzle vs LDS-pad bank-conflict tradeoff): https://github.com/iree-org/iree/issues/23765
- ROCm Blog — Max-Achievable FLOPs Part 2 (16×16 vs 32×32 power/clock): https://rocm.blogs.amd.com/software-tools-optimization/measuring-max-achievable-flops-part2/README.html
- MI300X workload optimization (LDS padding vs occupancy; fnuz fp8): https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/workload.html
- HipKittens (arXiv 2511.08083 — wave specialization underperforms on CDNA3/4): https://arxiv.org/abs/2511.08083
