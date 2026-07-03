# Q3VL MXFP4 TP1 — Session Log 2026-07-03
**Node: MI355X (gfx950) | Stack: vLLM a65093c + aiter 0.2.0 (flydsl 0.2.0) + patch 0006**
**Baseline: 9.22–9.24 QPS (ml-perf offline, real Shopify 8K, 4000 samples)**

---

## What was asked

Fix tile_m=128 stage1 (O1) and cshuffle stage2 (O2) in `/opt/aiter` to reach 10–12 QPS.

---

## O2 — cshuffle stage2 removal

**Status: Cannot be done via env var. Requires aiter source modification.**

- `FLYDSL_MOE_STAGE2_CSHUFFLE=0` → triggers `raise ValueError(...)` for f16/bf16 output
- `FLIR_MOE_STAGE2_CSHUFFLE=0` → same issue
- The FP4 stage2 path (`compile_mixed_moe_gemm2` in `mixed_moe_gemm_2stage.py`) hard-requires cshuffle at line ~2970:
  ```python
  if not _use_cshuffle_epilog:
      raise ValueError("stage2 f16 output currently requires CShuffle epilogue")
  ```
- **Fix requires**: implementing a direct-write epilogue branch in `compile_mixed_moe_gemm2()` analogous to stage1's `use_cshuffle=False` path

---

## O1 — tile_m=128 stage1 correctness bug

### What was confirmed working
- The existing aiter test `test_flydsl_moe_a4w4.py --stage stage1 --block-m 128` **passes** — but uses token≤1024 with E=256, TOPK=8, which never exercises WG bx≥1 (only 1 M-block per expert at those token counts with block_m=128)
- WG bx=0 (bx_m=0) always produces **correct output** for all shapes
- Baseline 9.24 QPS uses tile_m=64 via heuristic — NOT tile_m=128. The bug doesn't affect current baseline.

### Root cause investigation — what was found

**Symptom:** For tile_m=128, WG bx≥1 (bx_m=128+) always produces **exactly zero output**. Deterministic zeros. Reproducible across all token sizes, E counts, and TOPK values.

**Key isolation:** E=1, TOKEN=129 → WG bx=0 handles tokens 0–127 (correct), WG bx=1 handles token 128 (zero).

**What was ruled out:**
- LDS size overflow: total LDS = 98KB < 163KB gfx950 limit ✓
- Buffer OOB for sorted_rsrc: all accesses within bounds ✓  
- Buffer OOB for sx_rsrc (A scale): all accesses within bounds ✓
- Expert ID assignment: expert_ids[bx=1] correctly set ✓
- blk_valid check: bx_m=128 < num_valid[0]=256 → True ✓
- exp_valid check: expert_id < experts → True ✓
- X row stride (c_k_div4): correct at 512 dwords/row for fp4 ✓
- vmcnt undercount: partially ruled out (tile_m=64 has same undercount ratio yet works)

**Suspected mechanism (unverified):**
The `layout_x_tile_div4 = (tile_m, tile_k_dwords)` where `tile_k_dwords = tile_k // (4 * a_elem_vec_pack) = 32` for fp4. For tile_m=128 with num_x_loads=8:
- Loads i=0..3: row_raw = 0..127, within tile_m=128 ✓
- Loads i=4..7: row_raw = 128..255, idx2crd wraps modularly via `row_raw % tile_m`

When idx2crd wraps `row_raw=128 → row_mod=0`, the sorted_rsrc lookup becomes `bx_m + 0 = bx_m` (correct first row for the WG). However the LDS store for these loads targets wrapped rows 0..63, **overwriting** the data that loads i=0..3 already wrote. The overwritten data comes from X tokens at sorted_ids[bx_m+0..63] — for WG bx=1, these are the SAME tokens as loads i=0..1 for WG bx=0, not the tokens WG bx=1 should process. This corrupts the A tile in LDS before MFMA reads it.

**Attempted fix (WRONG — reverted):**
Changed `tile_k_dwords = tile_k * a_elem_bytes // (4 * a_elem_vec_pack)` → `tile_k * a_elem_bytes // 4` (removes the a_elem_vec_pack=2 divisor). This doubles the layout from (128, 32) to (128, 64), making i=4..7 map to rows 64..127 without wrap. But this did NOT fix the zero output — confirming the root cause is elsewhere.

### What needs to happen to fix O1

The actual fix requires **ISA-level debugging** to identify which instruction path for WG bx≥1 produces zeros:
1. Use `rocgdb` to set a breakpoint in the compiled flydsl kernel and inspect MFMA accumulator values for WG bx=1 vs bx=0
2. OR use `rocprofv3 --kernel-trace` on a small workload and diff the instruction traces for WG bx=0 vs bx=1
3. OR add Python-level `print` statements inside `compile_mixed_moe_gemm1()` to emit the MLIR IR, then inspect the generated ISA for bx_m-dependent branches

**The bug is definitely in `compile_mixed_moe_gemm1()`** in:
```
/opt/aiter/aiter/ops/flydsl/kernels/mixed_moe_gemm_2stage.py
```
Specifically in how bx_m affects either the X-tile loading, scale loading, or MFMA accumulation for blocks beyond the first M-tile block.

---

## Key facts for the next session

### Current state
- **Baseline**: 9.24 QPS (confirmed 2026-07-03, 4000 samples, 0 failed)
- **Active kernel at saturation (token=32768)**: `flydsl_moe1_afp4_wfp4_bf16_t64x128x256_w4_bnt0_fp4` + `flydsl_moe2_afp4_wfp4_bf16_t64x128x256_atomic` (heuristic, NOT from tuned CSV)
- **Tuned fmoe CSV**: `/tmp/aiter_configs/tuned_fmoe.csv` — has 405 fp4 per_1x32 entries but NONE matching Q3VL shape (cu=256, E=128, topk=8, model=4096, inter=1536)
- **fuse_quant intact**: `grep -c "Q3VL fuse_quant" /opt/aiter/aiter/fused_moe.py` = 2 ✓
- **No source changes made** (the tile_k_dwords change was reverted)

### Roofline ceiling if O1+O2 fixed
- Stage2 (19.8% GPU time, 3.9% efficiency): fixing cshuffle → ~30% efficiency → **+8–12% QPS**
- Stage1 (15.9% GPU time, 14.2% efficiency): fixing tile_m=128 → ~25% efficiency → **+8–15% QPS**  
- Combined potential: **10.6–13.3 QPS** (conservative to aggressive estimate)

### File locations
- Results doc: `/workspace/Hyperloom/q3vl_mxfp4_results.md` (Addenda 1–7)
- Offline uplift plan: `/workspace/Hyperloom/q3vl_mxfp4_offline_uplift_plan.md`
- Archive: `/workspace/Hyperloom/q3vl_mxfp4_experiments_20260701.tar.gz`
- Large files (traces): `/data2/hf_hub_cache/q3vl-mxfp4-experiments/`
- Model weights: `/tmp/q3vl-mxfp4`
- Benchmark configs: `/tmp/mlperf-base` (baseline, pristine)
- Key kernel source: `/opt/aiter/aiter/ops/flydsl/kernels/mixed_moe_gemm_2stage.py`

### The one thing that might work without ISA debugging
Looking at the dispatch log: the existing tuned_fmoe.csv entries that DO use tile_m=128 (for other models) — do those models work correctly in practice? If yes, the bug is specific to our (E=128, inter=1536, model=4096) shape interacting with something in the kernel initialization. The test passes with E=256, inter=256 — the difference might be in how sorted_m is computed or how the scale buffer is sized for this specific shape. Worth checking if any of the 405 CSV entries with tile_m=128 actually exercise bx≥1 (i.e., sorted_m > block_m for those models).
