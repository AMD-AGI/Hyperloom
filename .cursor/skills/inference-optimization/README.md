# Inference Optimization Skill

Closed-loop LLM inference optimization on AMD Instinct GPUs: profile with TraceLens, optimize kernels with GEAK, verify improvement, repeat.

| File | Description |
|------|-------------|
| `SKILL.md` | Main skill — phases 0-10: classify → baseline → profile → analyze → tune → GEAK → patch → sweep → report |
| `KNOWLEDGE-BASE.md` | Model-specific configs, validated results, pitfalls, benchmark fairness case studies |
| `GEAK-INFERENCE-KERNEL.md` | Deep reference — GEAK MCP details, kernel extraction, integration paths |
| `scripts/common.sh` | Shared functions — kill_server, wait_for_health, check_benchmark_lib, filter_trace, check_gpu_memory |
| `scripts/run_baseline.sh` | Baseline benchmark + profiling (single server launch) |
| `scripts/run_profile.sh` | Profiling run against an already-running server |
| `scripts/run_sweep.sh` | Parameter sweep (CONC/ISL/OSL grid with server reuse) |
| `README.md` | This file — usage guide |

### Output directories (on shared NFS)

```
/shared_nfs/inference-optimization/
├── results/<timestamp>/     # Benchmark JSON results + server logs
├── traces/<timestamp>/      # Profiler traces (directly on NFS for TraceLens)
```

## What It Does

1. Classify model architecture and select optimization strategy (Phase 0)
2. Run a baseline inference benchmark — SGLang or vLLM (Phase 1-2)
3. Profile the serving engine with torch.profiler (Phase 3)
4. Send trace to **TraceLens** for kernel-level bottleneck analysis (Phase 4)
5. Identify hot kernels that GEAK can optimize (Phase 5)
6. Tune server parameters — CUDA graph coverage, decode-steps, memory (Phase 6)
7. Submit kernels to **GEAK** for AI-driven optimization (Phase 7)
8. Patch optimized kernels, re-benchmark, **keep improvements, revert regressions** (Phase 8)
9. Sweep parameters (CONC, ISL/OSL) with the optimized version (Phase 9)
10. Generate report with optimization history, Pareto curves, and comparison (Phase 10)

## Prerequisites

- **GPU**: AMD Instinct MI355X / MI325X / MI300X (ROCm 7.0+)
- **Framework**: SGLang v0.5.9+ installed
- **InferenceX**: Cloned (e.g. `/shared_nfs/xiaofei/InferenceX`)
- **Model**: Downloaded locally (e.g. `/shared_nfs/xiaofei/models/DeepSeek-R1-0528`)
- **MCP servers**:
  - TraceLens — kernel profiling analysis
  - GEAK — kernel optimization

## Quick Start

### Example 1: Full optimization loop (RECOMMENDED — validated +14.4%)

```
@inference-optimization Optimize Qwen3-30B-A3B inference on MI355X.
Model: /shared_nfs/xiaofei/models/Qwen3-30B-A3B
InferenceX: /shared_nfs/xiaofei/InferenceX

Run the full torch.compile + GEAK pipeline:
1. Baseline (no torch.compile)
2. torch.compile + inductor baseline
3. Profile in torch.compile mode
4. TraceLens analysis to find top kernels
5. Extract Inductor-generated Triton kernels, submit to GEAK
6. Patch GEAK output into Inductor cache
7. Restart + E2E benchmark
Generate report.
```

This follows the **verified best path**: torch.compile generates Triton kernels → GEAK optimizes them → patch back into Inductor cache. Validated result: **571 → 653 tok/s (+14.4%)** on Qwen3-30B-A3B / MI355X.

### Example 2: Analysis only (no GEAK)

```
@inference-optimization Profile DeepSeek-R1 inference bottlenecks.
Model: /shared_nfs/xiaofei/models/DeepSeek-R1-0528
Just run baseline + TraceLens analysis. Show kernel breakdown.
No GEAK optimization or parameter sweep needed.
```

### Example 3: Sweep only (skip optimization)

```
@inference-optimization Sweep DeepSeek-R1 inference across CONC=4,8,16,32,64.
Model: /shared_nfs/xiaofei/models/DeepSeek-R1-0528
ISL/OSL: 1k/1k and 8k/1k. Skip TraceLens/GEAK, just benchmark.
```

## Reproducing the +14.4% E2E Result

**Validated on:** 2026-03-20, Qwen3-30B-A3B, 1x MI355X, SGLang v0.5.6

### Prerequisites

1. **Hardware**: AMD MI355X (gfx950) with ROCm 7.0+
2. **Model**: Qwen3-30B-A3B downloaded to local path
3. **SGLang**: v0.5.6+ with torch.compile support
4. **InferenceX**: Cloned for benchmark_serving.py
5. **MCP servers**: GEAK + TraceLens configured in `.cursor/mcp.json`

### What the agent does (8 steps, ~40 min)

```
Step 1 (~3 min): Launch SGLang baseline → benchmark → 571 tok/s
Step 2 (~9 min): Launch SGLang + torch.compile + inductor → benchmark → 600 tok/s (+5%)
Step 3 (~4 min): Profile with torch.profiler → generate trace
Step 4 (~1 min): Analyze trace → find RMSNorm kernel at 6.5% GPU time
Step 5 (~8 min): Extract kernel from Inductor cache → submit to GEAK → wait
Step 6 (<1 min): Download GEAK result (single-pass RMSNorm, 70% kernel speedup)
Step 7 (<1 min): Patch standalone kernel files in Inductor cache, clear binary cache
Step 8 (~2 min): Restart SGLang → benchmark → 653 tok/s (+14.4%)
```

### Key technical details

- **torch.compile is required**: Without it, SGLang uses aiter C++ kernels (no Triton for GEAK)
- **Inductor cache patching**: Only patch **standalone** kernel files (have `@triton_heuristics`, NO `async_compile`). Graph module files contain multiple kernels — patching them breaks other kernels.
- **Binary cache must be cleared**: Both `/tmp/torchinductor_root/**/*.so` and `~/.triton/cache/`
- **Server must restart**: New process recompiles patched .py → captures new CUDA graphs
- **Use `--mem-fraction-static 0.6`**: torch.compile needs extra memory during compilation; 0.7 causes OOM on 30B models

### Reference results

| Config | tok/s | TPOT | Gain |
|--------|-------|------|------|
| Baseline (SGLang, no compile) | 571.3 | 6.78ms | — |
| + torch.compile (inductor) | 600.1 | 6.44ms | +5.0% |
| + GEAK single-pass RMSNorm | 653.3 | 5.90ms | **+14.4%** |

## How It Differs from workload-optimization

| Aspect | workload-optimization | inference-optimization |
|--------|----------------------|----------------------|
| Scenario | Training | Inference serving |
| What gets modified | `workload.py` code | Kernels inside SGLang/vLLM |
| Benchmark tool | `harness.py` | InferenceX `benchmark_serving` |
| Optimization loop | Edit code → run harness → keep/revert | GEAK kernel → patch framework → restart server → benchmark → keep/revert |
| TraceLens role | Diagnose bottlenecks | Same — core of the loop |
| GEAK role | Optional kernel tuning | Core — optimize serving-path kernels |
| After optimization | Final report | Parameter sweep → Pareto curves → report |

## Output

- **`results.tsv`** — every optimization attempt with metrics and keep/discard status
- **Optimization report** (`.md`) — TraceLens analysis, GEAK results, Pareto curves, InferenceX comparison
- **Profiler traces** — baseline and optimized Chrome traces
- **Kernel backups** — original kernels in `$WORK_DIR/kernels/` for rollback

## Tips

- **Use torch.compile path for GEAK**: Without torch.compile, SGLang uses aiter C++ kernels (not Triton), giving GEAK no targets. With torch.compile, Inductor generates 293 Triton kernels — GEAK can optimize them.
- **DeepSeek-R1 on SGLang**: ~90% GPU time is vendor kernels — limited GEAK targets even with torch.compile
- **Qwen3-30B-A3B is a good test model**: Smaller, faster iteration, proven +14.4% E2E gain
- **TraceLens may fail on large traces (>300MB)**: Parse locally with Python instead (see Phase 4 in SKILL.md)
- **GEAK tasks take 5-30 min**: Fast if pod is warm (~5 min), slow on cold start (~30 min). Submit early.
- **Server restart with torch.compile takes ~3 min** (compilation overhead). Budget for this.
- **Always `unset PROFILE SGLANG_TORCH_PROFILER_DIR`** after profiling to avoid 30x slowdown
- **Only patch standalone Inductor kernel files** — graph module files contain multiple kernels, patching them causes `NameError`
