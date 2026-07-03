# Q3VL MXFP4 TP1 — Offline Uplift Plan
**Target: 10–12 QPS (current baseline: 9.22 QPS on MI355X gfx950)**
**Date: 2026-07-01 | Stack: vLLM a65093c + aiter patch 0006 + flydsl 0.2.0**

---

## 0. What was tried today (and why it didn't work)

| Experiment | Result | Root cause |
|---|---|---|
| Env sweeps (7 configs) | All flat/regressive | Scheduler already optimal; kernel bottleneck |
| FlyDSL tile sweep (microbench) | +10.4% at token=4096 | Token=4096 never dispatched in real serving |
| FlyDSL tuned CSV (e2e) | 9.17 QPS (−0.5%) | Tile hit wrong dispatch tier |
| `_reduce` stage2 mode | Server crash (HTTP 500) | Kernel produces garbage output (cos=0.007) |
| `tile_m=128` stage1 | Kernel produces garbage | Correctness bug in current aiter 0.2.0 |

**Core finding**: the current flydsl dispatch for `token=32768` (primary saturation tier) is already near-optimal within the *correct* kernel subset. The remaining 35.7% of GPU time in MoE GEMM is currently running at ~5–8% of FP4 peak TFLOPS — a large gap, but caused by architectural constraints that require kernel source changes, not config tuning.

---

## 1. Saturation-point profile (benchmark.py, Shopify 8K, max_throughput)

> Captured 2026-07-01 via `+server.profile=true`, 800 samples, ~840 concurrent reqs, KV 98%.
> All times are total GPU time across the 800-sample capture window.

### 1.1 Full kernel breakdown

| Rank | Kernel | CUDA time | % total | Calls | Time/call |
|---|---|---|---|---|---|
| 1 | `mfma_moe2_afp4_wfp4_bf16_cshuffle_t64x128x256` (stage2) | 9.44s | **19.8%** | 3407 | 2.77ms |
| 2 | `kernel_unified_attention_2d` (decode, ROCM_AITER_UNIFIED_ATTN) | 7.76s | **16.3%** | 3491 | 2.22ms |
| 3 | `FmhaFwdKernel` / `mha_varlen_fwd` (prefill FMHA, CK Tile) | 7.75s | **16.3%** | 882 | 8.78ms |
| 4 | `mfma_moe1_silu_mul_afp4_wfp4_fp4_t64x128x256_pm1_fp4` (stage1) | 7.56s | **15.9%** | 3407 | 2.22ms |
| 5 | `f4gemm_bf16_per1x32Fp4_BpreShuffle_256x256` (attn QKV/O proj) | 4.04s | **8.5%** | 6804 | 0.59ms |
| 6 | `rms_norm` (CK Tile Rmsnorm2dFwd) | 2.06s | **4.3%** | 6978 | 0.30ms |
| 7 | `aten::addmm` (vision encoder BF16, Tensile) | 2.04s | **4.3%** | 4680 | 0.44ms |
| 8 | `add_rmsnorm_quant_kernel` (residual+norm+FP8 quant) | 1.11s | **2.3%** | 6989 | 0.16ms |
| 9 | `elementwise_kernel_manual_unroll` (misc fused elem) | 0.99s | **2.1%** | 7953 | 0.12ms |
| 10 | `triton_poi_fused_1` (RoPE/misc triton) | 0.93s | **1.95%** | 3760 | 0.25ms |
| 11 | `dynamic_per_group_scaled_quant` (FP8 per-token quant) | 0.92s | **1.93%** | 11280 | 0.08ms |
| 12 | `static_per_tensor_quant` (FP8 KV quant) | 0.49s | 1.0% | 3760 | 0.13ms |
| 13 | `reshape_and_cache_flash` (KV cache fill) | 0.45s | 0.9% | 3760 | 0.13ms |
| 14 | `moe_sorting_fwd` + `mxfp4_moe_sort` | 0.44s | 0.9% | 7520 | 0.06ms |
| 15 | `topkGatingSoftmax` | 0.05s | 0.1% | 3760 | 0.01ms |

### 1.2 Bucket summary

```
MoE GEMM (stage1 + stage2)      35.7%   ← primary target
Attention (prefill + decode)     32.6%   ← secondary
Attention linear (FP4 GEMM)      8.5%   ← tertiary
Norm layers (rms + add_rms)      6.6%
Vision encoder BF16 GEMMs        4.3%   ← tertiary
Misc quant / elementwise         4.0%
Sort / routing / gating          1.0%
```

---

## 2. MI355X hardware specifications

| Parameter | Value |
|---|---|
| Architecture | gfx950 (MI355X) |
| Compute units | 256 CUs |
| FP4 MFMA peak (theoretical) | **2,614 TFLOPS** |
| BF16 MFMA peak | **1,307 TFLOPS** |
| HBM bandwidth | **6.4 TB/s** (8×HBM3e) |
| L2 cache | 128 MB |
| Ridge point (FP4) | 408 FLOP/byte |
| Ridge point (BF16) | 204 FLOP/byte |

---

## 3. Roofline analysis

### 3.1 MoE GEMM stage1 — flydsl `mfma_moe1_...t64x128x256`

**Shape at token=32768 (saturation):**
- M_sorted = 32768, N = 3072 (INTER×2, gate+up), K = 4096 (hidden)
- Per expert: M_expert = 32768/128 = **256 tokens avg**
- FP4 weights: 128 × 3072 × 4096 ÷ 2 = 805 MB
- FP4 input: 32768 × 4096 ÷ 2 = 64 MB
- BF16 output (fused quant→FP4): 32768 × 3072 × 0.5 = 48 MB (FP4 packed)

```
Arithmetic intensity = 2×32768×3072×4096 / (805+64+48)MB = 1024 FLOP/byte
Ridge point (FP4)   = 408 FLOP/byte
→ COMPUTE BOUND (AI >> ridge)
```

**Observed performance:**
```
Observed:  7.557s / 3407 calls = 2.218ms per call
FLOPs:     824.6 GFLOP per call
Obs TFLOPS: 372 GFLOP / 2.218ms = ~0.37 TFLOPS
Peak FP4:  2614 TFLOPS
Efficiency: 0.37 / 2614 = ~14.2% of FP4 peak     ← large gap
```

**Why so low despite being compute-bound?**
- Tile_m=64, M_expert=256 → only 4 M-tiles per expert
- 128 experts × 4 M-tiles × 24 N-tiles = 12,288 WGs launched
- 12,288 WGs / 256 CUs = 48 sequential waves per CU (fine)
- **Real culprit: per-group scale application in K-loop**
  - K=4096, group_size=32 → 128 scale loads per row
  - e8m0 scale tensor is `[M, K/32]` shape — accessed with stride, poor coalescing
  - Each of 128 K-tiles loads a scale, stalling MFMA pipeline
  - This is a memory-access pattern inefficiency, not a FLOP ceiling issue
- **Fused activation quant**: the `_fp4` suffix means stage1 fuses the output FP4 quantization inline — adds extra store traffic

**Roofline ceiling at 30% efficiency:** 2.218ms × (14.2%/30%) = **1.05ms** → **−1.17ms/call × 3407 calls = −4.0s saved**

---

### 3.2 MoE GEMM stage2 — flydsl `mfma_moe2_...cshuffle_t64x128x256`

**Shape at token=32768:**
- M_sorted = 32768, N = 4096 (hidden, output), K = 1536 (INTER)
- Weight: 128 × 4096 × 1536 ÷ 2 = 402 MB (FP4)
- Input: 32768 × 1536 ÷ 2 = 24 MB (FP4, from stage1 fused output)

```
Arithmetic intensity = 2×32768×4096×1536 / (402+24+256)MB = 1024 FLOP/byte
→ COMPUTE BOUND
```

**Observed performance:**
```
Observed:  9.440s / 3407 calls = 2.771ms per call
FLOPs:     412.3 GFLOP per call
Obs TFLOPS: 412.3 / 2.771 = ~149 GFLOPS = 0.149 TFLOPS
Peak FP4:  2614 TFLOPS
Efficiency: 0.149 / 2614 = ~5.7% of FP4 peak     ← extreme underperformance
```

**Why stage2 is even slower than stage1?**
- Stage2 has HALF the FLOPs of stage1 but takes 25% MORE time → 2.77ms vs 2.22ms
- The kernel name contains `cshuffle` (column-shuffle epilogue)
- **C-shuffle** is a layout transformation needed when the MFMA output register layout doesn't match the desired output tensor layout
- It requires: (1) writing output to LDS in MFMA layout, (2) reading back in transposed order, (3) writing to global memory
- This doubles the LDS bandwidth and adds an extra global store of the full output tile
- **Extra BW from cshuffle**: 32768 × 4096 × 2 bytes = 256 MB extra store pass
- At 6.4 TB/s: 256 MB / 6.4 TB/s = 0.04ms extra — small, but the LDS round-trip is latency-critical
- **Root cause**: the flydsl stage2 always uses cshuffle for `atomic` mode to handle the reduction correctly — but it may be avoidable with a direct-write epilogue for the non-atomic case

**Roofline ceiling at 30% efficiency:** 2.771ms × (5.7%/30%) = **0.526ms** → **−2.24ms/call × 3407 calls = −7.6s saved**

> **Combined MoE at 30%**: save ~11.6s out of 47.6s total GPU time → **+32% QPS ceiling ≈ 12.1 QPS**

---

### 3.3 Prefill FMHA — CK Tile `FmhaFwdKernel`

**Shape:** 882 prefill calls (of 1080 total attention ops), BF16, nheads=64, head_dim=128
- Variable seqlen from Shopify prompts (avg ~500 tok text + vision)
- Vision encoder ViT: nheads=16, head_dim=96, seqlen=4096 (64×64 ViT patches)

```
O(seqlen^2) FLOPs — memory bound for seqlen < 512, compute bound for seqlen > 1k
Observed: 7.749s / 882 calls = 8.78ms per call (includes multimodal image prefill)
```

**Assessment:** CK Tile FA2 is the best-in-class kernel for gfx950 (hand-optimized by AMD). No user-accessible tuning surface. The 16.3% share is fundamental to the multimodal workload.

---

### 3.4 Decode attention — `kernel_unified_attention_2d` (ROCM_AITER_UNIFIED_ATTN)

```
Observed: 7.763s / 3491 calls = 2.22ms per call
Shape: bs ≤ 512 (cudagraph), nheads=64, head_dim=128, paged KV cache (fp8_e4m3)
```

At bs=512 decode: purely memory-bound (reading all KV blocks). Already using the best backend (`ROCM_AITER_UNIFIED_ATTN` confirmed faster than `ROCM_AITER_FA` in earlier tests). No tuning opportunity.

---

### 3.5 Attention FP4 linear — `f4gemm_bf16_per1x32Fp4_BpreShuffle_256x256`

**Shape:** M ≤ 512 (cudagraph decode sizes), N=K=4096 (QKV + O projections)

```
At M=512: AI = 2×512×4096×4096 / (512×4096/2 + 4096×4096/2 + 512×4096×2) = ~341 FLOP/byte
Ridge (FP4) = 408 → marginally MEMORY BOUND at M=512
At M=256: AI = ~170 → MEMORY BOUND (well below ridge)
```

**Observed:** 4.037s / 6804 calls = 0.593ms per call
**Expected at HBM limit (M=512):** `(4096×4096/2 + 512×4096×4) / 6.4e12` ≈ 0.002ms → clearly launch-overhead dominated at small M

The `BpreShuffle_256x256` kernel is the preshuffled ASM GEMM path (`VLLM_ROCM_USE_AITER_FP4_ASM_GEMM=1` enables it). No better alternative exists in this build.

---

### 3.6 Vision encoder BF16 GEMM — `aten::addmm` (Tensile fallback)

**Shapes from startup log:**
```
M=131072, N=3456/1152/4304, K=1152/4304  (ViT self-attention projections)
M=38596,  N=1152/4304, K=1152/4304        (ViT cross-attention)
M=32768,  N=9216, K=4096                  (cross-attn QKV, a4w4 blockscale — separate path)
```

**Roofline for M=131072, N=1152, K=1152:**
```
FLOPs: 2×131072×1152×1152 = 347.9 GFLOP
Bytes: (131072+1152)×1152×2 + 131072×1152×2 ≈ 607 MB
AI: 573 FLOP/byte > BF16 ridge (204) → COMPUTE BOUND
Theoretical time: 347.9 GFLOP / 1307 TFLOPS = 0.27ms
Number of calls at saturation: ~4680 aten::addmm total, but vision is ~1 call per request
  → ~800 calls × 3-4 distinct shapes = ~3200 / 4680
Observed total addmm time: 2.043s / 4680 calls = 0.44ms each
```

**Gap:** Expected 0.27ms, observed 0.44ms → **62% efficiency** via Tensile (not terrible, but aiter could do better). The log shows "not found tuned config in `/tmp/aiter_configs/bf16_tuned_gemm.csv`" — these shapes have never been tuned for this workload.

---

## 4. Uplift opportunities — ranked by impact and effort

### Tier 1 — High impact, requires aiter kernel source work

#### **O1: Fix `tile_m=128` stage1 correctness bug** (potential: +15–20% QPS)

**What:** The flydsl FP4 stage1 kernels with `tile_m=128` produce cos~0.007 output (garbage) for all token tiers. They are registered in `_KERNEL_PARAMS` but functionally broken for the per_1x32 quantization path.

**Why it matters:**
- tile_m=128 would give 2× larger tiles → better MFMA pipeline fill
- At M_expert=256, tile_m=128 → only 2 M-tiles per expert (vs 4 for tile_m=64)
- Fewer but bigger tiles → better register reuse, less L2 pressure per WG
- Estimated: 30–50% kernel speedup if tile_m=128 worked correctly

**Root cause to investigate:**
- The `_fp4` fused-quant suffix on stage1 combines the MXFP4 activation quant into the MFMA epilogue
- At tile_m=128, the LDS allocation is 2× larger → may exceed LDS limit (64KB/CU on gfx950)
- Or: the fused e8m0 scale application at tile_m=128 has a wrong index calculation
- Location: `/opt/aiter/aiter/ops/flydsl/kernels/moe_gemm_2stage.py`, function `compile_moe_gemm1()`, the `_use_cshuffle_epilog` and LDS allocation logic

**How to fix:**
1. Add a correctness test: `test_flydsl_moe_a4w4.py --stage stage1 --block-m 128`
2. Inspect the LDS size calculation at line ~326: `lds_x_bytes = 2 * tile_m * lds_stride * elem_bytes` — at tile_m=128 this is 128KB which exceeds the 64KB LDS limit
3. Fix: either (a) reduce the ping-pong buffer to 1 buffer at tile_m=128, or (b) add a LDS size guard and split the K-loop differently
4. Alternatively: implement tile_m=128 without the fused `_fp4` epilogue (unfused is faster at large tiles anyway)

**Expected QPS impact:**
- Stage1: 15.9% of GPU time, currently 14.2% efficient. At 25% with tile_m=128: save ~6% e2e → **+0.55 QPS**
- Stage2: no change yet

---

#### **O2: Eliminate stage2 cshuffle epilogue** (potential: +8–12% QPS)

**What:** `mfma_moe2_afp4_wfp4_bf16_cshuffle_t64x128x256` takes 2.77ms vs stage1's 2.22ms despite only half the FLOPs. The `cshuffle` epilogue is the suspected cause.

**Why cshuffle exists in stage2:**
- The `atomic` accumulation mode reduces partial results from split-K workgroups
- To avoid race conditions, results are written to LDS first, then atomically accumulated
- The `cshuffle` reorders the output from MFMA register layout to row-major BF16

**How to eliminate:**
- Option A: Use `direct` epilogue (no cshuffle) when `k_batch=1` (no split-K) — this is already the case at token=32768 with heuristic ksplit=-1, but the kernel was compiled with cshuffle mandatory
- Option B: Add `use_cshuffle_epilog=False` path to `compile_moe_gemm2()` in `moe_gemm_2stage.py` — currently stage2 always uses the cshuffle path regardless of split-K
- Location: `moe_gemm_2stage.py`, look for `compile_moe_gemm2()`, parameter `use_cshuffle_epilog`
- The stage1 already has `use_cshuffle_epilog` as a parameter (line 107) — stage2 needs the same treatment

**Expected QPS impact:**
- If stage2 achieves the same ~14% efficiency as stage1 (2× speedup): save 1.4ms/call × 3407 calls = 4.8s = **+10% QPS** ≈ **+0.92 QPS**

---

#### **O3: Fix `_reduce` mode stage2** (potential: +3–5% QPS if used for high-ksplit)

**What:** `flydsl_moe2_..._reduce` kernels produce garbage output (cos~0.008). These are needed for ksplit>1 in stage2. Without working reduce, we can't safely use ksplit > 1 in stage2.

**Why reduce mode exists:** When K is split across multiple WGs, each WG computes a partial result that must be atomically reduced. The `reduce` mode handles this accumulation.

**Root cause to investigate:** The `_reduce` epilogue in `moe_gemm_2stage.py` uses bf16 atomic adds. On gfx950 (`gfx950` starts bf16 global atomics support according to `supports_bf16_global_atomics`), this should work. The bug may be in the reduction index calculation or the scale application ordering.

**Expected QPS impact:** Needed as a prerequisite for O4 below, not standalone.

---

### Tier 2 — Medium impact, achievable with tuning/config

#### **O4: K-split (k_batch) for stage1 tile_m=64** (potential: +3–6% QPS after O2/O3)

**What:** The registry has `t32x128x256_w3_kb2/4/7/14` kernels (k_batch > 1) but NO k_batch variants for tile_m=64. To use ksplit with tile_m=64, we need to either:
- Add tile_m=64 ksplit kernels to `get_flydsl_stage1_kernels()` in `moe_kernels.py`
- Or use tile_m=32 ksplit kernels (which exist, but have fewer MFMA units per WG)

**How to implement:**
1. In `moe_kernels.py`, `get_flydsl_stage1_kernels()`: add `k_batches=[1, 2, 4]` for `tile_m=64` (currently only `tile_m=32` gets `k_batches` > 1)
2. Recompile/register the new kernel variants
3. Add tuned FMOE CSV rows for token=32768 using these new variants
4. Validate correctness first (the tile_m=64 base kernel is correct, ksplit just adds K-dim parallelism)

**Expected QPS impact:** ksplit=2 doubles WGs → better SM occupancy at low M_expert. At 30% efficiency improvement: **+2–4% QPS**

---

#### **O5: Vision encoder BF16 GEMM tuning via aiter tuner** (potential: +1–2% QPS)

**What:** Vision encoder GEMM shapes are all using Tensile fallback (torch generic). The aiter BF16 GEMM tuner can find per-shape-optimal kernel configs.

**Shapes to tune:**
```python
shapes = [
    (131072, 3456, 1152),  # ViT QKV proj (largest, most important)
    (131072, 1152, 1152),  # ViT attention proj
    (131072, 4304, 1152),  # ViT FFN gate+up
    (131072, 1152, 4304),  # ViT FFN down
    (38596,  4608, 4608),  # cross-attn sizes
    (38596,  4096, 4608),
]
```

**How to run:**
```bash
# Build untuned CSV
python3 -c "
shapes = [(131072,3456,1152),(131072,1152,1152),(131072,4304,1152),(131072,1152,4304),(38596,4608,4608),(38596,4096,4608)]
print('M,N,K,bias,dtype,outdtype,scaleAB,bpreshuffle')
for m,n,k in shapes:
    print(f'{m},{n},{k},True,torch.bfloat16,torch.bfloat16,False,False')
" > /tmp/vision_untuned.csv

ROCR_VISIBLE_DEVICES=2 python3 /opt/aiter/csrc/ck_gemm_tuner/gemm_tune.py \
    -i /tmp/vision_untuned.csv \
    -o /tmp/vision_tuned.csv \
    --iters 100 --warmup 20

# Merge into AITER_CONFIG_GEMM
cp /tmp/aiter_configs/bf16_tuned_gemm.csv /tmp/bf16_tuned_gemm_v2.csv
cat /tmp/vision_tuned.csv >> /tmp/bf16_tuned_gemm_v2.csv

# Run offline with env var pointing at new CSV
# Add to benchmark.yaml env block:
# AITER_CONFIG_GEMM: "/tmp/bf16_tuned_gemm_v2.csv"
```

**Expected QPS impact:** Observed 0.44ms vs theoretical 0.27ms at 62% efficiency. Tuning could reach 75–80% → save ~0.2ms/call × 4680 calls = 0.93s = ~**+2% QPS** ≈ +0.18 QPS

---

#### **O6: XCD swizzle for stage1 kernel** (potential: +1–3% QPS)

**What:** The registry has `_xcd4` variants of all stage1/stage2 kernels (XCD swizzle = cross-CU data distribution swizzle). These haven't been tested.

**What xcd4 does:** Reorders the WG-to-tile assignment so that spatially adjacent WGs map to different XCDs (cross-chiplet dies on MI300X/MI355X, which has 6 GCDs each with multiple XCDs). This improves L2 cache utilization by ensuring weight tiles for different experts are distributed across XCDs.

**How to test:** Simply add `_xcd4` suffix to the existing winning kernel in the tuned CSV:
```
kn1 = "flydsl_moe1_afp4_wfp4_bf16_t64x128x256_w4_bnt0_fp4"  # current
kn1 = "flydsl_moe1_afp4_wfp4_bf16_t64x128x256_w4_bnt0_fp4_xcd4"  # try
kn2 = "flydsl_moe2_afp4_wfp4_bf16_t64x128x256_atomic"  # current
kn2 = "flydsl_moe2_afp4_wfp4_bf16_t64x128x256_atomic_xcd4"  # try
```

Note: correctness MUST be validated before e2e bench (the non-xcd4 base kernels were correct; xcd4 only changes scheduling, not compute, so should be correct too — but verify).

**Expected QPS impact:** On MI300X workloads, XCD swizzle typically gives 2–5% on compute-bound MoE. Since stage1+2 = 35.7%: **+0.7–1.8% e2e** ≈ +0.06–0.16 QPS

---

### Tier 3 — Low impact, worth trying if Tier 1/2 gaps are closed

#### **O7: Block-size 16 for paged attention KV read** (potential: +0.3–0.8% QPS)

Current: default block_size (16 or 32). The `paged_attention_ll4mi_QKV` kernel is in the decode path but already using ROCM_AITER_UNIFIED_ATTN. Block-size affects KV read coalescing.

```bash
# In tp1_mxfp4.yaml, add:
block_size: 16
```

Need to verify `reshape_and_cache_flash` doesn't regress.

---

#### **O8: Fuse `dynamic_per_group_scaled_quant` into MoE sort** (potential: +0.5% QPS)

`dynamic_per_group_scaled_quant` at 1.93% is already fused into the flydsl stage1 epilogue (that's what `_fp4` suffix means). The remaining 11280 calls are for the attention FP8 KV path (`aiter::dynamic_per_group_scaled_quant`). These are CK kernels — no easy fusion surface.

---

## 5. Sequence and expected cumulative QPS

| Step | Action | Prerequisite | Expected delta | Cumulative QPS |
|---|---|---|---|---|
| Baseline | Current (`t64x128x256`, heuristic) | — | — | **9.22** |
| O6 | Test `_xcd4` variants for stage1+2 | none, 1 hour | +0 to +2% | 9.22–9.40 |
| O5 | Vision encoder BF16 GEMM tuning | aiter GEMM tuner run | +1–2% | 9.31–9.58 |
| O4 | Stage1 k_batch=2 for tile_m=64 (if O3 not needed) | New kernel variant | +2–4% | 9.50–9.96 |
| O2 | Remove cshuffle from stage2 | aiter kernel source fix | +8–12% | 10.26–11.1 |
| O1 | Fix tile_m=128 correctness | aiter kernel source fix | +8–15% | 11.1–12.8 |
| O3 | Fix `_reduce` mode + ksplit | aiter kernel source fix | +2–4% | 11.3–13.3 |

**Conservative path (O6+O5+O2 only):** 9.22 × 1.02 × 1.02 × 1.10 ≈ **10.6 QPS (+15%)**
**Aggressive path (all):** 9.22 × 1.02 × 1.02 × 1.10 × 1.12 × 1.12 ≈ **13.3 QPS (+44%)**

---

## 6. Where to look in the codebase

### Stage1/2 flydsl kernel source
```
/opt/aiter/aiter/ops/flydsl/kernels/moe_gemm_2stage.py
  └── compile_moe_gemm1()   line ~93   — tile_m=128 LDS size bug lives here
  └── compile_moe_gemm2()              — cshuffle epilogue to disable

/opt/aiter/aiter/ops/flydsl/moe_kernels.py
  └── get_flydsl_stage1_kernels()  line ~69  — add k_batch variants for tile_m=64
  └── get_flydsl_stage2_kernels()  line ~143 — check cshuffle control
  └── _KERNEL_PARAMS dict           — registry where new kernels must be added
```

### Dispatch/heuristic
```
/opt/aiter/aiter/fused_moe.py
  └── use_mxfp4_flydsl block  line ~1420  — token tier → kernel name heuristic
  └── get_2stage_cfgs()       line ~960   — CSV lookup (override heuristic)
  └── tuned_fmoe.csv          /tmp/aiter_configs/tuned_fmoe.csv — inject here
```

### Vision encoder GEMM tuning
```
/opt/aiter/csrc/ck_gemm_tuner/gemm_tune.py  — GEMM tuner
/tmp/aiter_configs/bf16_tuned_gemm.csv       — tuned GEMM lookup
```

### LDS size verification
```python
# In compile_moe_gemm1():
lds_x_bytes = 2 * int(tile_m) * int(lds_stride) * int(elem_bytes)
cshuffle_bytes = _cshuffle_elem_bytes * int(tile_m) * int(tile_n)
total_lds = lds_x_bytes + cshuffle_bytes
# For tile_m=128, tile_n=128, elem_bytes=0.5 (fp4), lds_stride=256+pad:
# lds_x_bytes = 2 * 128 * 256 * 0.5 = 16384 bytes (fine, < 64KB)
# cshuffle_bytes = 2 * 128 * 128 = 32768 bytes
# total = 49152 bytes — fits in LDS (64KB limit)
# So the LDS size is NOT the problem → look elsewhere
```

The bug is more likely in the **fused FP4 quant epilogue** at tile_m=128. When the gate+up outputs are packed into FP4 with e8m0 scales, the index calculation for the scale tensor may overflow or alias incorrectly at tile_m=128 vs tile_m=64.

**Specific debugging approach:**
```bash
# Run the existing a4w4 test with verbose output at tile_m=128
ROCR_VISIBLE_DEVICES=3 python3 /opt/aiter/aiter/ops/flydsl/test_flydsl_moe_a4w4.py \
    --stage stage1 --block-m 128 -t 32768 --verbose

# Compare output tiles element-by-element to find the corruption pattern
# If it's a striding issue: the error pattern will have a period = tile_m
# If it's a scale issue: errors will cluster at group boundaries (every 32 elements)
```

---

## 7. Key invariants to preserve

1. **`AITER_ONLINE_TUNE=0` always** — online tuning crashes MXFP4 serving (allocates wrong tile)
2. **fuse_quant patch 0006 always active** — verify `grep -c "Q3VL fuse_quant" /opt/aiter/aiter/fused_moe.py` = 2 before any aiter change
3. **Validate correctness before e2e bench** — use cos similarity ≥ 0.99 vs heuristic reference; tile_m=128 and `_reduce` mode currently fail this gate
4. **Score on `benchmark.py` with real Shopify dataset, 4000 samples** — synthetic `vllm bench serve` results don't correlate (confirmed: +10% synthetic = flat e2e)
5. **Stage2 cshuffle**: when disabling cshuffle, use `accumulate=True` (default atomic add) for correctness; `accumulate=False` (overwrite) is only valid for ksplit=1 where there's no reduction
6. **FP4_ASM_GEMM=1 and CK_MOE_SORTING=1** already in baseline env — don't remove
7. **tile_m=32 kernels (all variants)** are known correct and should be used for ksplit testing before tile_m=64 ksplit variants are built

---

## 8. Validation gate checklist per iteration

```bash
# 1. Kernel correctness (run on GPU3, ~30s)
ROCR_VISIBLE_DEVICES=3 python3 /tmp/flydsl_tile_bench4.py  # cos check mode

# 2. Coherence check (run after server starts, ~1 min)
curl -s http://localhost:8050/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"/tmp/q3vl-mxfp4","messages":[{"role":"user","content":"Why is the sky blue?"}],"max_tokens":50}' \
  | python3 -c "import sys,json; r=json.load(sys.stdin); print(r['choices'][0]['message']['content'])"
# Expected: non-repeating, coherent Rayleigh-scattering explanation

# 3. Log check (after server starts)
grep "AITER_MXFP4_MXFP4\|Q3VL fuse_quant" /tmp/off-XXX/outputs/tuned/vllm_server.log

# 4. Full offline benchmark (GPU0, ~10 min)
cd /tmp/off-XXX && ROCR_VISIBLE_DEVICES=0 ... python benchmark.py \
  server=tp1_mxfp4 scenario=offline_qwen3_vl_235b_a22b_shopify \
  model=/tmp/q3vl-mxfp4
grep 'QPS:' outputs/XXX/.../endpoints_client.log

# Target: ≥ 9.22 QPS (baseline) with 0 failed samples
```

---

## 9. Quick reference: actual dispatch keys at saturation

During `benchmark.py` offline with 4000 Shopify samples, fused_moe is dispatched with these `token` values (observed from serve log):

| Token | When | Current kernel |
|---|---|---|
| 1–512 | CUDA graph decode (cudagraph_capture_sizes) | `t32x128x256_w2_fp4` + `t32x128x256_atomic_bnt2` |
| 1024–2048 | Early batch warmup | `t32x128x256_w2_fp4` + `t32x128x256_atomic_bnt2` |
| 8192 | Mid-batch prefill | `t64x128x256_w4_bnt0_fp4` + `t64x128x256_atomic` |
| 16384 | Large prefill batch | `t64x128x256_w4_bnt0_fp4` + `t64x128x256_atomic` |
| **32768** | **Steady-state saturation (primary)** | **`t64x128x256_w4_bnt0_fp4` + `t64x128x256_atomic`** |

> Any tuning effort must target **`token=32768`** to impact offline QPS.
> Token=4096 is absent from the real serving dispatch and has no QPS impact.

---

## 10. Summary judgment

| Kernel | % GPU time | Current efficiency | Roofline ceiling | Hand-tunable? | Expected gain |
|---|---|---|---|---|---|
| MoE stage2 (cshuffle) | 19.8% | 5.7% FP4 peak | 30%+ with direct epilogue | **Yes — remove cshuffle** | +8–12% QPS |
| MoE stage1 (tile_m=64) | 15.9% | 14.2% FP4 peak | 25%+ with tile_m=128 fix | **Yes — fix correctness bug** | +8–15% QPS |
| Prefill FMHA | 16.3% | ~best-in-class | CK Tile optimal | No | 0% |
| Decode attention | 16.3% | memory-bound, near-optimal | HBM ceiling | No | 0% |
| Attn FP4 linear (M≤512) | 8.5% | launch-overhead at small M | Hard without bigger batch | Indirect only | 0% |
| Vision BF16 GEMM | 4.3% | ~62% BF16 peak | 80%+ with tuned kernel | **Yes — run GEMM tuner** | +1–2% QPS |
| Norms + quant | 6.3% | memory-bound, acceptable | Near-optimal | No | 0% |
| MoE sort/routing | 1.0% | acceptable | — | No | 0% |

**The path to 10–12 QPS runs through the flydsl MoE kernel source**:
- Stage2 cshuffle removal: highest ROI, moderate effort (Python kernel code)
- Stage1 tile_m=128 fix: highest ceiling, requires finding and fixing the epilogue bug
- Both are in `/opt/aiter/aiter/ops/flydsl/kernels/moe_gemm_2stage.py`

---

# ADDENDUM (2026-07-03, opus session) — MEASURED RESULTS SUPERSEDE §4 AND §10

Everything above (§3-§10) was written 2026-07-01 from roofline *estimates* before any of the
levers were measured e2e. This addendum records what actually happened when they were tested.
**Where this addendum conflicts with §4/§10, this addendum wins.**

Baseline this session (clean solo, GPU0, same host): **9.04 QPS** (historical 9.22-9.24; ~2%
host variance). All comparisons below are solo, 4000 Shopify samples, 0 failed unless noted.

## MoE-kernel levers — ALL CLOSED (see results.md Addenda 8-9)

| Lever (§ ref) | Estimated | MEASURED | Verdict |
|---|---|---|---|
| O1 fix tile_m=128 stage1 (§4 O1, §10) | +8-15% | **not a bug; tile_m=128 correct but ~1.8x SLOWER** | dead |
| O2 remove stage2 cshuffle (§4 O2, §10) | +8-12% | stage2 already ~48% FP4 peak¹; marginal + uncoalesced-store risk | not built |
| 1-stage ASM fused MoE (fmoe_g1u1) | — (new) | correct + coherent but **7.67 vs 9.04 (~15% slower)** | reverted |

¹ The §3.2 "5.7% FP4 peak" figure for stage2 was a mis-estimate. Re-measured: stage2 at
token=32768 = 2.61ms for ~3.3 TFLOP fp4 ≈ 1265 TFLOPS ≈ **48% of the 2614 peak**. Stage2 is
NOT the huge underperformer §3.2/§10 claimed; the epilogue tail is small, so O2's ceiling is
~+2-4% at best, with real regression risk (cshuffle exists to coalesce scattered MFMA lanes).

**O1 detail**: the §4-O1 / §10 "tile_m=128 produces garbage (cos 0.007)" claim was an artifact
of a buggy forced-dispatch microbench. Verified correct via the official test_flydsl_moe_a4w4.py
(stage1/stage2/e2e all pass at Q3VL dims, bx≥1 exercised) and the fused-fp4 serving path (98.97%
code-match, zero all-zero rows). It is correct but slower — the larger tile loses MFMA-pipeline
efficiency at this saturation shape. **Do not re-investigate O1.**

## Attention front — MEASURED, all closed (this session, task #36)

§3.3-§3.5 asserted attention was untunable "0%" without measuring. Now measured:

| Lever | MEASURED | Verdict |
|---|---|---|
| ROCM_AITER_FA backend (offline op point) | **7.82 QPS** (−13%) | ✗ regression — UNIFIED_ATTN wins at offline too |
| ROCM_AITER_UNIFIED_ATTN (baseline) | 9.04 QPS | ✓ optimal, keep |
| VLLM_ROCM_USE_AITER_TRITON_ROPE=1 | **8.98 QPS** (flat) | ✗ RoPE is 1.95% + already efficient |
| VLLM_ROCM_SHUFFLE_KV_CACHE_LAYOUT=1 | n/a | ✗ consumed ONLY in rocm_aiter_fa.py (FA-only), no-op with UNIFIED_ATTN |
| VLLM_ROCM_FP8_MFMA_PAGE_ATTN=1 | n/a | ✗ gates paged_attention_rocm (ll4mi/ROCM_ATTN path), not UNIFIED_ATTN's fused kernel |

Why FA loses even at the offline (prefill-heavy, large-batch) operating point: FA uses SEPARATE
kernels — aiter ck_tile mha_varlen for prefill + paged_attention_v1 for decode
(rocm_aiter_fa.py:871/1344) — whereas UNIFIED_ATTN runs one fused kernel_unified_attention_2d for
both, which wins on this mixed chunked-prefill+decode batch. The earlier mc=32 result
(results.md:36, FA 6.07 < UA) now confirmed to also hold at offline. **Attention is genuinely at
its practical ceiling on this build — the "0%" in §3.3-§3.5 was right, now verified.**

The decode/KV knobs (SHUFFLE_KV, FP8_MFMA_PAGE_ATTN) are structurally coupled to the FA / paged
backends and cannot help the UNIFIED_ATTN configuration. Switching backend to unlock them costs
more (−13%) than the knobs could ever recover.

## What actually remains worth trying (revised, honest)

| Lever | Type | Realistic gain | Status |
|---|---|---|---|
| **O5 Vision-encoder BF16 GEMM tuning** | config-only (aiter a16w16 tuner + CSV) | +1-2% | **UNTESTED — best remaining** |
| **O6 XCD swizzle (_xcd4 MoE kernels)** | config-only (tuned CSV suffix) | 0 to +2% | **UNTESTED — cheap** |
| O7 paged-attn block_size=16 | config-only | +0.3-0.8% | untested, tiny, needs backend change → likely net-neg |

O5 mechanism confirmed viable: vision GEMMs (N,K ∈ {1152,3456,4304,4608,4096}) all hit
"not found tuned config → torch fallback" (~62% peak). tuned_gemm.py keys on **padded_M** (buckets
variable per-request M), so a tuned row generalizes across runtime M. Tuner:
`/opt/aiter/csrc/gemm_a16w16/gemm_a16w16_tune.py`; CSV: `/tmp/aiter_configs/bf16_tuned_gemm.csv`
(AITER_CONFIG_GEMM_BF16_FILE). Note §6's path `csrc/ck_gemm_tuner/gemm_tune.py` is WRONG — use
csrc/gemm_a16w16/gemm_a16w16_tune.py.

Everything else (O3 reduce-mode, O4 k_batch tile_m=64) requires kernel source/compilation and
only feeds the already-dead MoE-tile path — not worth it given O1/O2 are closed.

## Bottom-line roadmap (replaces §4 sequence table and §10)

- Config-only levers realistically sum to **~+2-4% → ~9.2-9.4 QPS**, NOT 10-12.
- **10-12 QPS is not reachable by kernel *selection* or env tuning on this build.** It needs
  either (a) a NEW/retuned MoE kernel — larger stage2 tile OR a fused kernel with block_m≥64
  (the compiled 1-stage ASM is block_m=32 and loses) — authored/tuned for the Q3VL shape, or
  (b) scale-out (8× TP1 replicas, the actual MLPerf system layout), which is an infra change not
  a single-GPU kernel win.
- Recommended next action if pursuing single-GPU: run O5 (vision GEMM tune) + O6 (xcd4) as a
  cheap batch, bank the ~2-4%, and escalate the MoE-kernel authoring to the aiter team with the
  measured evidence that all existing dispatchable kernels are exhausted.
