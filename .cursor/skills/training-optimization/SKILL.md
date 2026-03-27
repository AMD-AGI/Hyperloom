---
name: workload-optimization
description: Iterative, closed-loop GPU training optimization. Takes a real distributed training workload (Primus/Megatron stack), profiles with torch.profiler and TraceLens (mandatory), diagnoses bottlenecks, then enters an optimization loop where the agent applies config overrides, code patches, or GEAK kernel-level rewrites one at a time, measures ms/iter from actual training, keeps improvements, reverts regressions, and iterates until no further gains. Global batch size must remain constant throughout — only config/code efficiency improvements are valid. Use when the user asks to optimize a workload, speed up training, reduce iteration time, or find bottlenecks.
---

# GPU Training Optimization — Iterative Closed-Loop

## Overview

This is an **agent-driven optimization loop** for real distributed training workloads. You (the agent) are the loop controller. You will:

1. Set up and run the training workload to establish a baseline
2. Profile the workload and diagnose bottlenecks
3. **Enter an optimization loop**: think → apply one change → measure → keep/revert → repeat
4. *(Optional)* Send hot custom kernels to **GEAK** for AI-driven kernel-level optimization
5. Write a final report

The loop continues until you've tried at least 8 ideas or 3 consecutive ideas fail to improve.

**Key principle:** You run *actual training* as the benchmark. No separate harness or synthetic micro-benchmarks — the real `torchrun` command with real (or mock) data is both the workload and the measurement tool.

### Live Knowledge Base Contribution (MANDATORY)

**You MUST update the "Knowledge Base: Hard-Won Lessons" section at the bottom of this file in real time as you work — not at the end, DURING the optimization loop.** When you encounter any of the following, immediately append to the appropriate subsection:

- **New pitfall**: A config override that crashes, regresses, or has a non-obvious constraint — append to "What to Avoid" with the exact error and context.
- **Validated result**: After a KEEP/DISCARD decision that reveals something non-obvious (e.g., a config that helps on one model but hurts another), append with exact numbers.
- **Workaround**: Any non-obvious fix (port conflicts, compilation issues, environment setup).
- **GEAK outcome**: Kernel name, micro-bench speedup, E2E result, keep/revert.
- **New config override discovery**: If you find a config flag that produces >0.5% improvement, add it to "Config Override Priority" with the model and gain %.

**Rules for writing:**
- **Append-only during a run** — do not reorganize or rewrite existing entries. Just add under the right heading.
- **Always include a validation date**: `*(Validated YYYY-MM-DD on <hardware>.)*`
- **Be specific**: Exact numbers, exact error messages, exact commands. Vague notes are useless to future runs.
- **Write for your future self**: Another agent instance will read this. What would have saved you 30 minutes?

**When NOT to write**: Don't add entries for things already in the Knowledge Base section. Read it first. Only add genuinely new information.

## Setup Phase

### Step 1: Understand the training stack

Read the user's training configuration to understand:
- **Config file** (YAML): model architecture, parallelism (TP, PP, EP), precision, batch size, sequence length
- **Launch script**: how `torchrun` is invoked, what overrides are supported
- **CLI argument system**: how config overrides are passed (e.g., `key=value` appended to the command)

For Primus/Megatron workloads, the typical structure is:
```
/workspace/Primus/
├── examples/megatron/configs/<GPU>/  # YAML configs
├── examples/run_pretrain.sh          # Launch wrapper
├── primus/cli/main.py                # CLI entry point
└── primus/core/utils/arg_utils.py    # Override parsing
```

Overrides are typically key=value pairs appended after the `--config` argument:
```bash
torchrun --nproc_per_node=8 --master_port=29500 \
  -m primus.cli.main train pretrain \
  --config path/to/config.yaml \
  key1=value1 key2=value2
```

### Step 2: Establish baseline

Run the training for a fixed number of iterations (typically 10) and extract ms/iter from the training log output.

```bash
torchrun --nproc_per_node=<NUM_GPUS> --master_port=29500 \
  -m primus.cli.main train pretrain \
  --config <CONFIG_YAML> \
  profile=false use_pytorch_profiler=false \
  2>&1 | tee /tmp/baseline.log
```

**Measurement protocol:**
- Use **iterations 6–10** for timing (skip 1–5 for warmup, JIT compilation, NCCL init)
- Extract `elapsed time per iteration (ms)` from the training log
- Compute the average — this is your **baseline ms/iter**
- If the log format differs, look for `throughput`, `samples/sec`, `TFLOP/s`, or similar metrics

Record the baseline. This is the number to beat.

### Step 3: Profile and diagnose bottlenecks

Re-run with profiling enabled to collect a PyTorch Chrome trace:

```bash
torchrun --nproc_per_node=<NUM_GPUS> --master_port=29500 \
  -m primus.cli.main train pretrain \
  --config <CONFIG_YAML> \
  profile=true use_pytorch_profiler=true \
  profile_step_start=6 profile_step_end=7 \
  2>&1 | tee /tmp/profile.log
```

This produces a `.pt.trace.json` file. Analyze the trace to understand the kernel breakdown:

```python
import json

with open("<trace_file>") as f:
    trace = json.load(f)

gpu_events = [e for e in trace["traceEvents"]
              if e.get("cat") == "kernel" and "dur" in e]
gpu_events.sort(key=lambda e: -e["dur"])

# Group by kernel name and compute total time
from collections import defaultdict
kernel_time = defaultdict(float)
kernel_count = defaultdict(int)
for e in gpu_events:
    kernel_time[e["name"]] += e["dur"]
    kernel_count[e["name"]] += 1

total = sum(kernel_time.values())
for name, t in sorted(kernel_time.items(), key=lambda x: -x[1])[:20]:
    print(f"  {name[:70]:70s}  {t/1000:>8.1f}ms  {t/total*100:>5.1f}%  {kernel_count[name]:>4d}x")
```

Identify:
- What fraction is GEMM (vendor BLAS — `Cijk_*` kernels)? These are hard to improve.
- What fraction is attention? Which backend (aiter, Triton, CK)?
- What fraction is MoE dispatch/permutation?
- What fraction is communication (NCCL)?
- Any obvious overhead (generic PyTorch kernels like `CatArrayBatchedCopy`, `scatter_gather_elementwise`)?

#### REQUIRED: TraceLens analysis

TraceLens analysis is **mandatory** — run it on every baseline and final-optimized trace. TraceLens provides hierarchical GPU timeline breakdowns, roofline modeling, and communication analysis that the basic kernel profile cannot. Do NOT skip this step.

TraceLens is accessed via the **`oci-traceLens-agent` MCP server** configured in `.cursor/mcp.json`. Use its MCP tools directly — do NOT attempt to call TraceLens via CLI or hardcoded filesystem paths.

**Running TraceLens via MCP:**

Use the `run_full_standalone_analysis` MCP tool from the `oci-traceLens-agent` server:

```
Tool: run_full_standalone_analysis
Arguments:
  trace_path: <path to .pt.trace.json>
  platform: "MI355X"       (must be uppercase)
  trace_type: "pytorch"
  output_dir: <results directory>/tracelens_output
  cleanup: false
```

Run this on BOTH the baseline trace and the final optimized trace. Store outputs in separate subdirectories (e.g., `tracelens_output/baseline/` and `tracelens_output/optimized/`).

To directly compare two traces, use the `run_comparative_analysis` tool:

```
Tool: run_comparative_analysis
Arguments:
  gpu1_kineto: <path to baseline trace>
  gpu2_kineto: <path to optimized trace>
  gpu1_name: "baseline"
  gpu2_name: "optimized"
  cleanup: false
```

You can also use `check_trace_file` to verify a trace path exists before running analysis.

**Available MCP tools (oci-traceLens-agent):**

| Tool | Purpose |
|------|---------|
| `check_trace_file` | Verify trace path exists, get file size |
| `run_full_standalone_analysis` | Full analysis pipeline on one trace |
| `run_comparative_analysis` | Compare two traces (baseline vs optimized) |

**Notes:**
- `platform` must be uppercase, one of: `MI300X`, `MI325X`, `MI350X`, `MI355X`, `MI400`
- Trace files must be on NFS-accessible paths (the MCP server reads them directly)
- No GPU required — TraceLens is pure CPU analysis

### Step 3b: Optional — Check for GEAK kernel optimization candidates

After profiling, scan the top-20 GPU kernels for GEAK candidates. GEAK uses an AI agent to rewrite kernel source code (Triton or HIP) for better performance on the target hardware.

**When to use GEAK:**

| Kernel type in profile | GEAK applicable? | Why |
|------------------------|-------------------|-----|
| `Cijk_*` (hipBLASLt GEMM) | **No** | Vendor BLAS — already hand-tuned for MFMA |
| `aiter::fmha_v3_*` | **No** | Vendor attention — already optimized for gfx950 |
| `triton_*` / `triton::` / `_permute_kernel` | **Yes** | Triton kernels have Python source GEAK can rewrite |
| Custom HIP kernels (`__global__`) | **Yes** | Primary GEAK target |
| `vectorized_elementwise_kernel` chains | **Maybe** | Try `torch.compile` first; if still hot, GEAK |

**Decision rule:** If a kernel is in the top-5 by GPU time, has modifiable source code, and is NOT vendor BLAS or vendor attention, it's a GEAK candidate.

If you find candidates, read the full GEAK skill for the detailed flow:
`@.cursor/skills/workload-optimization/GEAK-KERNEL-OPTIMIZATION.md`

GEAK tasks take 10–30 minutes (GPU pod scheduling + agent steps). **Kick them off early** and continue the config/code optimization loop in parallel. When GEAK results arrive, integrate them as a normal optimization attempt — benchmark, keep or revert.

### Step 4: Initialize results log

Create `results.tsv` in the user's results directory:

```
attempt	ms_per_iter	speedup_pct	status	description
0	13265.3	0.0	baseline	GPT-OSS 20B BF16 baseline (8 GPU, EP=8, mock data, iter 6-10 avg)
```

## Critical Constraints

### Global Batch Size is IMMUTABLE

**The global batch size (GBS) must remain identical to the baseline throughout the entire optimization sweep.** Changing GBS changes the amount of work per iteration, making ms/iter comparisons meaningless — a 2× faster iteration that does half the work is not an optimization.

What is allowed:
- **Micro batch size (MBS)**: Can be changed (e.g., MBS=4 vs MBS=8) as long as GBS stays the same. This changes the number of gradient accumulation (GA) steps per iteration: `GA = GBS / (MBS × num_GPUs)`.
- **Gradient accumulation steps**: Automatically adjusts when MBS changes to maintain the same GBS.

What is **NOT** allowed:
- Reducing GBS (e.g., from 512 to 256) to make iterations faster. This is an **illegal optimization** — it reduces tokens processed per step.
- Any configuration change that alters the effective GBS, even indirectly.

**Enforcement:** Before marking any attempt as "keep", verify that `global_batch_size` in the training log matches the baseline GBS exactly. If it doesn't, mark the attempt as "discard (invalid — GBS changed)" regardless of the ms/iter result.

**When tracking BEST_OVERRIDES:** Only update BEST_OVERRIDES from attempts that maintain the same GBS as the baseline. Never let an invalid-GBS attempt become the "best" — this corrupts all subsequent attempts and the final profile.

## Optimization Loop

Now enter the closed loop. For each iteration:

### 1. THINK

Look at:
- `results.tsv` — what you've tried, what worked, what didn't
- The kernel profile — where is time being spent?
- The config YAML — which optimization flags are disabled by default?
- The training stack source code — are there cached lookup patterns, unfused operations, or suboptimal defaults?

Then decide what to try. **Make ONE change at a time** so you know what helped.

#### Types of changes

**Config overrides** (safest — just append to the torchrun command):
- `moe_permute_fusion=true` — fused Triton permute kernels for MoE token dispatch
- `gradient_accumulation_fusion=true` — fuses wgrad GEMM with optimizer accumulation
- `moe_use_fused_router_with_aux_score=true` — fused TopK router
- `use_turbo_grouped_mlp=true` — fused SwiGLU activation (may regress with wide FFN dims)
- `use_sink_attention=true/false` — toggle between Triton and aiter attention backends
- `sink_sliding_window=N` — sliding window attention (check backend support first)

**Code patches** (more invasive — edit source files directly):
- Cache expensive per-forward lookups (e.g., `get_args()`, config flag checks)
- Rewrite MoE token dispatch (pre-sorting, batched operations)
- Swap attention backends programmatically
- Add custom fused kernels (Triton or HIP)

**GEAK kernel optimization** (optional — kernel-level rewrites via AI agent):
- Submit hot Triton/HIP kernel source to GEAK MCP with shapes, dtype, and hardware context
- GEAK's LLM agent rewrites the kernel for the target GPU (e.g., better block sizes, vectorized loads, pipelining)
- Retrieve the optimized kernel, patch it in, and benchmark as a normal attempt
- See `GEAK-KERNEL-OPTIMIZATION.md` for the full MCP tool sequence (`geak_create_task` → `geak_submit_task` → poll → `geak_get_outputs`)

**Environment variables** (apply via export before torchrun):
- `PYTORCH_TUNABLEOP_ENABLED=1` — GEMM autotuning (rarely helps >0.5%, can be very slow)
- `NCCL_ALGO`, `NCCL_PROTO`, `NCCL_MIN_NCHANNELS` — communication tuning

### 2. TRY

Apply the change:

**For config overrides**, just add to the command:
```bash
torchrun --nproc_per_node=<NUM_GPUS> --master_port=<PORT> \
  -m primus.cli.main train pretrain \
  --config <CONFIG_YAML> \
  <existing_kept_overrides> <new_override>=<value> \
  profile=false use_pytorch_profiler=false \
  2>&1 | tee /tmp/attempt_N.log
```

**For code patches**, edit the source file, then run with all kept overrides.

**Important:** If a previous run left processes on the port, increment `--master_port`:
```bash
pkill -9 -f "primus/cli/main.py"  # kill lingering processes
# then use --master_port=29501, 29502, etc.
```

### 3. MEASURE

Extract ms/iter from the log (iterations 6–10 average), same protocol as baseline.

### 4. DECIDE

- **If ms_per_iter improved**: KEEP the change. Add the override to your running set (or keep the code patch). Log to results.tsv with status `keep`.
- **If ms_per_iter same or worse**: REVERT the change (remove the override or revert the code edit). Log to results.tsv with status `discard`.
- **If it crashed**: Log with status `crash`, revert. Note the error — it may reveal backend limitations.
- **If it took excessively long** (e.g., TunableOp tuning): Kill and log with status `cancelled`.

### 5. REPEAT

Go back to step 1. Use the updated results.tsv and kernel profile to pick the next idea.

### Stopping Criteria

The loop must stop at the right time. Use ALL of these checks after every attempt.

**1. Diminishing returns** (primary signal)
Track cumulative speedup. If the last 5 attempts produced < 0.5% total new improvement, you're in a plateau. Stop.

**2. Time budget**
Set a wall-clock limit at the start (default: 60 minutes for distributed training, since each run takes 3–5 min). Check elapsed time after each attempt.

**3. Crash budget**
Max 2 crashes total. After that, the environment is likely unstable. Stop and report.

**4. Consecutive failure limit**
3 consecutive discards (no improvement). You've exhausted the easy ideas at this abstraction level.

**5. Theoretical ceiling awareness**
From the initial profile, compute the GEMM fraction. If GEMMs are >60% and near-optimal (vendor BLAS), the maximum gain from config/code changes is limited. If you've already captured >50% of the non-GEMM headroom, further gains require fundamentally different approaches (FP8, architectural changes). Note this in the report and stop.

**6. "Good enough" threshold**
If total speedup exceeds 5% for config-only changes or 10% for code+config, that's a strong result. Report it.

**Summary decision table:**

| Condition | Action |
|-----------|--------|
| Last 5 attempts < 0.5% total gain | Stop — plateau |
| Wall clock > 60 min | Stop — time budget |
| 2+ crashes | Stop — unstable |
| 3 consecutive discards | Stop — local minimum |
| Total speedup > 10% | Stop — good enough |
| User interrupts | Stop — always |

**Always write the report even if stopped early — partial results are valuable.**

## After the Loop

### Re-profile the final optimized version

Run training one more time with profiling enabled and the **best equal-workload optimizations only**.

**CRITICAL:** The overrides used here must be the best config that keeps the **same GBS/MBS as baseline** (or same GBS with a different MBS if that was a valid kept change). Do NOT blindly use the last "keep" attempt's overrides — verify GBS matches baseline before profiling. If the last "keep" changed GBS, use the most recent "keep" that preserved GBS instead.

```bash
torchrun --nproc_per_node=<NUM_GPUS> --master_port=<PORT> \
  -m primus.cli.main train pretrain \
  --config <CONFIG_YAML> \
  <best_equal_workload_overrides> \
  profile=true use_pytorch_profiler=true \
  profile_step_start=6 profile_step_end=7 \
  2>&1 | tee /tmp/final_profile.log
```

Compare the final kernel breakdown against the baseline to understand where the gains came from.

### REQUIRED: Run TraceLens on final traces

Run TraceLens via the `oci-traceLens-agent` MCP server. Use `run_full_standalone_analysis` on both traces individually, then use `run_comparative_analysis` to get a direct comparison:

1. Analyze baseline trace → output to `$RESULTS_DIR/tracelens_output/baseline/`
2. Analyze optimized trace → output to `$RESULTS_DIR/tracelens_output/optimized/`
3. Run `run_comparative_analysis` with both trace paths to get a side-by-side comparison

The comparative analysis highlights per-category GPU time deltas, kernel-level regressions/improvements, and utilization changes.

**This step is mandatory.** The optimization report is incomplete without TraceLens analysis.

### Write the Report

Write to the user's results directory (e.g., `/shared_nfs/nehaprakriya/results/<workload>/optimization_report.md`):

```markdown
# <Workload> Optimization Report — <GPU> <N>-GPU

**Date:** YYYY-MM-DD
**Platform:** N× AMD <GPU>
**Container:** `<image>`
**Commit:** `<hash>`
**Model:** <description>

---

## Executive Summary

| Metric | Baseline | Optimized | Delta |
|---|---|---|---|
| **ms / iter** | X | Y | **−Z ms** |
| **Throughput** | A TFLOP/s/GPU | B TFLOP/s/GPU | **+C%** |
| Kept optimizations | — | M of N | |
| Discarded / crashed | — | K of N | |

**Final config overrides:**
```
override1=value1
override2=value2
```

**Code patches kept:** <description>

---

## All Attempts

| # | ms/iter | Δ vs baseline | Status | Description |
|---|---|---|---|---|
| 0 | X | — | baseline | ... |
| 1 | Y | +Z% | keep | ... |
| ... |

## What Worked
- ...

## What Didn't Work
- ...

## GEAK Kernel Optimization (if used)
| Kernel | GPU Time % | GEAK Task ID | Result |
|--------|-----------|--------------|--------|
| ... | ... | ... | keep/discard |

## Kernel Profile Comparison
(baseline vs final top-20 kernels)

## TraceLens Analysis
(include GPU timeline breakdown, ops summary by category, and key findings from TraceLens reports — this section is REQUIRED)

## Recommendations for Production
- ...

## Reproducibility
**Baseline command:**
```bash
torchrun ...
```
**Optimized command:**
```bash
torchrun ... <overrides>
```
```

## CI/CD Integration (Optional)

If the training stack has a CI/CD pipeline (e.g., `ROCm/unified-training-dockers`), you can submit optimizations for validation on CI hardware:

1. **Understand the CI structure**: Find the benchmark script that launches training in CI (e.g., `primus_megatron-lm_benchmark_report.sh`).
2. **Apply optimizations in the benchmark script**:
   - For config overrides: append them to the training command in the script.
   - For code patches: add `sed` commands before the training command to patch source files.
3. **Push to a feature branch** and create a PR against the CI branch (e.g., `pipeline`).
4. **Compare CI results** against the baseline CI run.

**Note:** CI hardware may differ from local hardware (e.g., MI325X in CI vs MI355X locally). Performance deltas may not transfer 1:1 across GPU architectures. Always note which hardware the optimizations were validated on.

## Knowledge Base: Hard-Won Lessons

These are things the agent has learned from prior optimization runs. Use them. **Update this section during the optimization loop when you discover something new** (see "Live Knowledge Base Contribution" in Overview).

### Config Override Priority (What to Try First)

Based on validated results from GPT-OSS 20B on MI355X:

1. **`moe_permute_fusion=true`** — largest single win (+1.19%). Replaces generic PyTorch MoE dispatch kernels with fused Triton permute kernels. Almost always helps for MoE models.
2. **`gradient_accumulation_fusion=true`** — reliable +0.46%. Fuses wgrad GEMM with accumulation. No downside.
3. **`moe_use_fused_router_with_aux_score=true`** — neutral perf but reduces kernel count. Free cleanup.
4. **Cache per-forward config lookups** — minor (+0.05%) but free. Look for `get_args()`, `use_split_wgrad_op()`, or similar calls in hot `forward()` methods.

### What to Avoid

- **Reducing `global_batch_size`** — this is NEVER a valid optimization. It reduces work per iteration, making ms/iter comparisons invalid. GBS must always match baseline. See "Critical Constraints" section.
- **`use_turbo_grouped_mlp=true`** with wide FFN dims — the fused SwiGLU kernel's tile config is suboptimal for non-square shapes, causing regression.
- **`sink_sliding_window`** — the Triton attention backend does not support sliding window. Crashes with `ValueError`.
- **`PYTORCH_TUNABLEOP_ENABLED=1`** during benchmarking — GEMM autotuning can take >30 min on MoE models with many unique shapes. Only useful as an offline pre-tuning pass.
- **Pre-sorting MoE tokens with `argsort + index_select`** when fused Triton permute is available — the fused kernel reads each token once and scatter-writes to all expert positions; `index_select` reads the same token N times (once per expert assignment).
- **Fusing gate+up into one wide GEMM** — bad hipBLASLt tiling for wide matrices.
- **`TunableOp + torch.compile` together** — TunableOp's runtime kernel probing crashes CUDA graph replay.

### aiter vs Triton Attention

Don't assume vendor-optimized (aiter) is always faster. For GPT-OSS 20B with `use_sink_attention=true` (64 Q-heads, 8 KV-heads, head_dim=128, seq=4096), the Triton attention backend was faster than aiter v3 native. Always measure.

### aiter `deterministic` Flag (CRITICAL)

`aiter.flash_attn_func` defaults to `deterministic=True`. On gfx950 with seqlen > 256, this DISABLES `fmha_v3_bwd` and falls back to legacy `mha_bwd` (2.7× slower BWD). PrimusTurbo defaults to `deterministic=False`. Always check what production uses.

### Port Conflicts After Killing Runs

If you kill a training run, the master port (default 29500) may stay bound. Increment the port for the next run (`--master_port=29501`). Always `pkill -9 -f "primus/cli/main.py"` before retrying.

### Mock Data Compilation

The Megatron-LM `MockGPTDataset` depends on a C++ helper that must be compiled first:
```bash
make -C /workspace/Primus/third_party/Megatron-LM/megatron/core/datasets
```
If mock data generation fails with a RuntimeError about building helpers, run this.

### hipBLASLt GEMMs

Default kernel selection is near-optimal for standard shapes. These `Cijk_*` kernels dominate GPU time (typically 60–70%) and cannot be improved through code changes. Gains come from reducing everything else.

### TraceLens (MANDATORY)

TraceLens analysis must be run on both the baseline and final optimized traces. It is NOT optional.

- Use the **`oci-traceLens-agent` MCP server** — do NOT hardcode paths to other users' directories
- MCP tool: `run_full_standalone_analysis` with `trace_path`, `platform`, `trace_type`, `output_dir`, `cleanup`
- No GPU required — TraceLens is pure CPU analysis
- `platform` must be uppercase: `"MI355X"` not `"mi355x"`
- Large traces (>30MB): use `trace_path` pointing to an NFS-accessible location

### Profiler Kernel Name Guide

| Kernel pattern | What it is |
|----------------|-----------|
| `Cijk_Ailk_Bljk_*` | hipBLASLt GEMM (NN layout) |
| `Cijk_Ailk_Bjlk_*` | hipBLASLt GEMM (NT layout) |
| `Cijk_Alik_Bljk_*` | hipBLASLt GEMM (TN layout) |
| `Custom_Cijk_*` | Custom hipBLASLt variant |
| `aiter::fmha_v3_fwd` | aiter v3 attention forward (fast) |
| `aiter::fmha_v3_bwd` | aiter v3 attention backward (fast, needs deterministic=False) |
| `aiter::mha_bwd` | Legacy aiter attention backward (slow, deterministic=True fallback) |
| `FmhaBwdDQDKDVKernel` | CK attention backward kernel (legacy) |
| `vectorized_elementwise_kernel` | Generic elementwise — candidate for fusion |
| `CatArrayBatchedCopy` | Generic copy kernel — sign of unfused MoE dispatch |
| `scatter_gather_elementwise` | MoE token gather/scatter — replaced by fused permute |
| `indexFuncLargeIndex` | MoE token indexing — replaced by fused permute |
| `_permute_kernel` | Fused Triton MoE permute (from moe_permute_fusion) |
| `_unpermute_kernel` | Fused Triton MoE unpermute (from moe_permute_fusion) |
| `aten::nonzero` | MoE routing overhead |
| NCCL kernels | Distributed communication (AlltoAll, AllReduce, etc.) |
