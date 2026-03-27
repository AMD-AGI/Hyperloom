# GLM-5-FP8 Optimization Resume State
## Saved: 2026-03-26 ~19:30 UTC

## GOAL
Push per-GPU throughput higher for `zai-org/GLM-5-FP8` on 8x MI355X GPUs.
Focus on code changes, kernel backends, and unique ideas — not just config sweeps.
The skill is at `/shared_nfs/nehaprakriya/agentic-rc/.cursor/skills/inference-optimization`

## BASELINE (TP=4, single instance)
- **1403.43 total tok/s** (per-GPU: 350.86)
- Config: TP=4, ds=8, CONC=64, ISL=1024, OSL=1024, tilelang NSA, no fused-moe-sum-allreduce
- DP=2/TP=4 previously hit 2794 tok/s but that's horizontal scaling, not per-GPU improvement

## COMPLETED RESULTS (all TP=4, CONC=64 unless noted)

| Experiment | Total tok/s | Delta | Notes |
|---|---|---|---|
| baseline (ds=8) | 1403.43 | — | Reference |
| fused_decode_mla=1 | 1396.28 | -0.51% | Worse |
| aiter_allreduce_fusion | 1406.83 | +0.24% | Neutral |
| tuned_fusedmoe_ds16 | 1408.5 | +0.4% | Tuned GEMMs + fused-moe flag (flag has NO effect on aiter path!) |
| tuned_fusedmoe_ds16_conc128 | FAILED | OOM | cuda-graph-max-bs 128 + conc 128 crashes |
| tuned_fusedmoe_ds32 | FAILED | crash | ds=32 server crashes during benchmark |
| tuned_fusedmoe_ds64 | 1413.6 | +0.7% | Higher decode steps help slightly |
| tuned_moe_triton_ds16 | 1411.0 | +0.5% | Triton MoE similar to aiter CK |
| **tuned_mixedchunk_ds16** | **1443.6** | **+2.9%** | **Mixed chunk scheduling WIN** |
| **tuned_nsa_aiter_ds16** | **1446.4** | **+3.1%** | **NSA aiter decode backend WIN** |
| tuned_mem90_ds16 | 1408.0 | +0.3% | Higher mem fraction neutral |
| tuned_allreduce_fusedmoe_ds16 | IN PROGRESS | — | Was running when node killed |

## STILL RUNNING WHEN NODE DIED
- `batch_v3.sh` was at experiment 9 of 12 (allreduce_fusedmoe_ds16)
- Remaining experiments from batch_v3.sh:
  - EXP 10: tuned_nccl32ch_ds16 (NCCL_MIN_NCHANNELS=32)
  - EXP 11: tuned_fusedmoe_ds32_conc128 (ds=32 + conc=128)
  - EXP 12: tuned_kitchen_sink_ds16 (allreduce fusion + fused moe + mixed chunk combined)

## KEY FINDINGS

### What works:
1. **--nsa-decode-backend aiter** (+3.1%) — switch NSA decode from tilelang to aiter
2. **--enable-mixed-chunk** (+2.9%) — overlap prefill/decode scheduling
3. These are ADDITIVE and should be combined

### What doesn't work / neutral:
- `--enable-fused-moe-sum-all-reduce` has **NO EFFECT on ROCm aiter path** (confirmed from code). The aiter `fused_moe` handles the topk reduction internally. Only affects Triton MoE path.
- `SGLANG_ROCM_FUSED_DECODE_MLA=1` — slightly worse
- `--enable-aiter-allreduce-fusion` — only +0.24% (fuses allreduce with RMSNorm+quant, too small to matter)
- Higher `mem-fraction-static 0.90` — neutral (0.85 already sufficient for 64 concurrency)
- Triton vs aiter MoE backend — essentially equivalent

### Critical bottleneck:
- Profiling showed **49% GPU idle, 44.9% NCCL communication**
- Only ~6% actual compute. The model is communication-bound at TP=4.
- 78 layers × 2 all-reduces/layer = 156 NCCL all-reduces per forward pass
- Piecewise CUDA graphs (which could overlap compute/comm) are disabled on ROCm
- SBO (Single Batch Overlap) only works with CUDA FlashInfer/DeepGemm backends

## TUNED KERNELS (ALREADY MERGED INTO AITER CONFIGS)

### Dense GEMM tuning (aiter a8w8_blockscale):
- Tuned 16 shapes for GLM-5 dense MLP: M=[1,2,4,8,16,32,64,128], N=6144, K={6144,3072}
- Merged into `/sgl-workspace/aiter/aiter/configs/a8w8_blockscale_tuned_gemm.csv` AND `/tmp/aiter_configs/`
- Source CSV: `glm5_tuned_gemm.csv`
- Impact: Minimal (only 3 dense layers out of 78)

### Fused MoE tuning (aiter fmoe):
- Tuned 11 shapes: token=[1,2,4,8,16,32,64,128,256,512,1024], model_dim=6144, inter_dim=512, expert=257, topk=9
- Merged into `/sgl-workspace/aiter/aiter/configs/tuned_fmoe.csv` AND `/tmp/aiter_configs/`
- Source CSV: `glm5_tuned_fmoe.csv`
- QuantType: per_1x128 (FP8 blockscale), fp8_e4m3fn dtype
- **NOTE**: For small batch (token*topk <= 128), aiter bypasses tuned configs for fp8 blockscale (line 794 in aiter/fused_moe.py)

## NEXT STEPS (PRIORITY ORDER)

### 1. COMBINE WINNERS (highest priority)
Run experiment combining ALL winning optimizations:
```bash
EXPERIMENT="combined_best" \
EXTRA_SERVER_ARGS="--nsa-decode-backend aiter --enable-mixed-chunk --num-continuous-decode-steps 16" \
bash rapid_experiment.sh
```
Expected: +3-6% from combining NSA aiter + mixed-chunk + ds=16

### 2. COMBINE WITH DS=64
```bash
EXPERIMENT="combined_ds64" \
EXTRA_SERVER_ARGS="--nsa-decode-backend aiter --enable-mixed-chunk --num-continuous-decode-steps 64" \
bash rapid_experiment.sh
```

### 3. NCCL/RCCL TUNING (untested)
Test RCCL environment variables to reduce the 44.9% NCCL overhead:
```bash
EXTRA_ENV="export NCCL_MIN_NCHANNELS=32; export NCCL_ALGO=Ring" \
```
Also try: NCCL_ALGO=Tree, NCCL_MIN_NCHANNELS=16/64, RCCL_MSCCL_ENABLE=1

### 4. CODE-LEVEL CHANGES (deeper investigation needed)
- The bypass at line 794 of `/sgl-workspace/aiter/aiter/fused_moe.py` skips tuned FMoE configs for small batches with FP8 blockscale. Consider removing bypass to use tuned configs always.
- Investigate custom all-reduce implementation for MI355X
- Try `--enable-mscclpp` flag for optimized collective communication
- Look at attention-layer GEMM shapes that are untuned: N=128, N=2624, K=4096 (MLA projections)

### 5. DP=2 + TP=4 WITH ALL OPTIMIZATIONS
Once per-GPU perf is maximized, run DP=2/TP=4 with all wins combined:
```bash
# Use the InferenceX launch script with DP=2, TP=4, and best server args
```

## FILE LOCATIONS
- Results dir: `/shared_nfs/nehaprakriya/agentic-rc/yanyuan_runs/glm5_optimization/results_v3/`
- Scripts dir: `/shared_nfs/nehaprakriya/agentic-rc/yanyuan_runs/glm5_optimization/`
- Rapid experiment runner: `rapid_experiment.sh`
- Batch experiment v3: `batch_v3.sh`
- Model path: `/shared_nfs/nehaprakriya/agentic-rc/yanyuan_runs/InferenceX/models/zai-org/GLM-5-FP8/`
- InferenceX dir: `/shared_nfs/nehaprakriya/agentic-rc/yanyuan_runs/InferenceX/`
- SGLang source: `/sgl-workspace/sglang/python/sglang/srt/`
- Aiter source: `/sgl-workspace/aiter/`
- Aiter tuned configs: `/sgl-workspace/aiter/aiter/configs/` (tuned_fmoe.csv, a8w8_blockscale_tuned_gemm.csv)
- Skill: `/shared_nfs/nehaprakriya/agentic-rc/.cursor/skills/inference-optimization/SKILL.md`

## KEY CODE PATHS
- MoE on ROCm+aiter: `fp8.py:1734` → `aiter.fused_moe(QuantType.per_128x128)` — bypasses Triton entirely
- FMoE config lookup: `aiter/fused_moe.py:675` (get_2stage_cfgs) — reads tuned_fmoe.csv
- FP8 blockscale bypass: `aiter/fused_moe.py:794` — skips tuned for token*topk<=128
- Allreduce fusion: `communicator.py:636` — requires enable_aiter_allreduce_fusion flag
- NSA decode backend: set via `--nsa-decode-backend aiter` (default is tilelang)
- Mixed chunk: set via `--enable-mixed-chunk`

## ENVIRONMENT ON NEW NODE
After node restart, you'll need to:
1. Verify tuned configs still exist in `/sgl-workspace/aiter/aiter/configs/` (they should, it's in the container image)
2. If `/tmp/aiter_configs/` is gone (it's tmpfs), the server auto-copies from the aiter source directory
3. Kill any leftover sglang processes: `pkill -9 -f sglang`
4. The rapid_experiment.sh and batch_v3.sh scripts are on shared_nfs and will survive node restart
