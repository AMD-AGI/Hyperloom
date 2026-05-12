# Action: Vendor Kernel Configuration

**DFS role:** Scores 5 for MoE+MLA models. This fills the gap when torch.compile is
incompatible and source-level OOB kernel optimization has limited targets. Vendor kernel dtype/fused/sort changes target
aiter kernels that dominate 40%+ of compute on these models. Apply sub-actions
sequentially; each is independently scored and can be pruned.

Configure and toggle vendor-provided kernel variants (aiter, CK, hipBLASLt) without
rewriting kernel source. This is distinct from `kernel-opt.md` (OOB Codex/Claude
rewrite source-backed kernels/code paths) — this action selects among **existing** vendor kernel paths
and configuration options.

## When to Use

This action is relevant when:
- kernel profile shows significant time in `aiter::*`, `ck::*`, `hipBLASLt::*`, or
  `hipModuleLaunchKernel::*` kernels
- Model uses vendor MoE kernels (`fmoe_fp8_blockscale_g1u1`, `moe_sorting_fwd`)
- Model uses FP8 weights or KV cache (dtype variant selection applies)
- `kernel-opt` scored low because torch.compile is unavailable or hot kernels are compiled/vendor-only

**Key insight:** For MoE+MLA models where source-level OOB optimization has few
targets, vendor kernel config can still yield 2–5% from dtype/fused/sort changes.
Use this action when the fastest path is selecting existing vendor implementations
rather than rewriting kernels.

## Inputs

- kernel profile (category breakdown with kernel names and efficiency metrics)
- Current server config and environment variables
- KB entries for this model's vendor kernel compatibility

## KB Query

```bash
python3 $SKILL_ROOT/kb/kb_query.py "$MODEL_NAME vendor kernel dtype MoE fused sort aiter" --top-k 10 --compact
python3 $SKILL_ROOT/kb/kb_query.py --category vendor_kernel_config --compact
```

## Sub-Actions

### VKC-1: FP8 Dtype Selection (fnuz vs OCP)

AMD GPUs support two FP8 formats:
- **fnuz** (`e4m3fnuz`): AMD-native, default on ROCm
- **OCP** (`e4m3`, "non-AI optimized"): industry-standard, used by NVIDIA

The aiter library may use different kernel paths depending on which format is active.
Switching can yield **~1.7% uplift** (validated on DSR1-class models).

**How to test:**
```bash
# Check current dtype
python3 -c "import torch; print(torch.float8_e4m3fn, torch.float8_e4m3fnuz)"

# Environment variable to force OCP format (framework-dependent)
export AMDGCN_USE_OCP_FP8=1  # or framework-specific flag
# Restart server, re-benchmark
```

**Interactions with MTP/speculative decoding:** Low risk — dtype change applies uniformly
to both base and draft forward passes. But SGLANG_USE_ROCM700A may select kernel paths
that assume a specific dtype. Test with MTP config, not baseline.

**accuracy_risk:** 0.15 (precision change — gate required)

### VKC-2: MoE Fused Kernel Toggle

The aiter library provides fused MoE kernels that combine sort + quantize + GEMM into
fewer dispatches. These may not be enabled by default or may have newer variants.

Known variants:
- `aiter::fmoe_fp8_blockscale_g1u1` — current default (sort + 2-phase GEMM)
- `aiter::fmoe_bf16_blockscaleFp8_g1u1_novs_silu` — BF16 input variant
- Newer fused variants may combine sort phases with GEMM dispatch

**Expected uplift:** ~1.5% (validated micro-benchmark, needs E2E verification)

**How to test:**
```bash
# Check available fused MoE implementations
python3 -c "import aiter; print([x for x in dir(aiter) if 'fmoe' in x or 'fused_moe' in x])"

# Environment variable or config to select variant
export AITER_FUSED_MOE_VERSION=2  # framework-dependent
# Restart server, re-benchmark
```

**Interactions with MTP:** Medium risk — MTP changes effective batch sizes during
draft/verify phases. Fused kernels may be tuned for specific batch size ranges. Test
with MTP config at multiple concurrency levels (CONC=4, 32, 64 minimum).

**accuracy_risk:** 0.10 (compute path change — gate required)

### VKC-3: GEMM Tuning CSV Selection

aiter and hipBLASLt use pre-tuned GEMM configurations stored in CSV files. Different
CSV files are tuned for different model shapes and batch sizes.

**How to check:**
```bash
# Find loaded tuning CSVs in server log
grep -i "tuned\|tuning\|csv" $RESULT_DIR/server.log | head -20

# Available tuning files
find /sgl-workspace/aiter -name "*tuned*.csv" -o -name "*tuning*.csv" 2>/dev/null
ls /opt/rocm/lib/hipblaslt/data/ 2>/dev/null
```

**Expected uplift:** 1–3% if current CSV is for a different model shape.

**accuracy_risk:** 0.05 (only affects GEMM tile selection, not computation)

### VKC-4: aiter Environment Flags

Environment variables that select kernel code paths within aiter:

| Flag | Effect | When to Use |
|------|--------|-------------|
| `SGLANG_USE_AITER=1` | Enable aiter kernels | Always (already default) |
| `SGLANG_USE_ROCM700A=1` | MI355X-specific optimizations | MTP config, MI355X only |
| `AMDGCN_USE_BUFFER_OPS=0` | Disable buffer ops (vLLM) | When buffer ops cause issues |
| `VLLM_ROCM_USE_AITER=1` | Enable aiter in vLLM | vLLM framework |
| `VLLM_ROCM_USE_AITER_TRITON_ROPE=1` | Use aiter Triton RoPE | vLLM with aiter |

**accuracy_risk:** 0.05 (code path selection)

## Procedure

### Step 1: Profile-Informed Triage

Read kernel profile output to identify vendor kernel categories and time distribution:

```
If MoE kernels (fmoe_*, moe_sorting_*) > 20% of compute:
    → VKC-1 (dtype), VKC-2 (fused) are relevant
If GEMM kernels (Cijk_*, a8w8_blockscale) > 15% of compute:
    → VKC-3 (GEMM tuning) is relevant
If aiter kernels are in use:
    → VKC-4 (env flags) — check for untested flags
```

### Step 2: Check KB for Known Regressions

```bash
python3 $SKILL_ROOT/kb/kb_query.py "$MODEL_NAME vendor kernel regression" --top-k 5 --compact
```

**CRITICAL:** If KB shows a known regression for a sub-action (e.g., MoE sort 3x
slower), **skip that sub-action** unless KB shows the regression was resolved after
a specific date/version.

### Step 3: Apply Sub-Actions Sequentially

Apply one sub-action at a time. After each:
1. Restart server (kill + wait + relaunch)
2. Benchmark at CONC=64 (quick single-point)
3. If gain > 0%: keep, move to next sub-action
4. If gain ≤ 0%: revert, log, move to next sub-action

**Order:** VKC-1 → VKC-2 → VKC-3 → VKC-4

### Step 4: Combined Verification

After all kept sub-actions are applied together:
1. Full concurrency sweep (CONC=4,8,16,32,64)
2. Accuracy gate (GSM8K) — required since accuracy_risk > 0

### Step 5: MTP Compatibility Check

If the model uses MTP/speculative decoding:
1. Re-run the full test with MTP config (not just baseline)
2. Compare MTP + vendor config vs MTP alone
3. Variable batch sizes from speculation can interact with kernel tuning

**If MTP config regresses but baseline improves:** Log as MTP-incompatible in KB.
The sub-action may still be useful for non-MTP deployments.

## Outputs

- Per sub-action results: (sub_action, gain_pct, status, config_change)
- Combined gain after all kept sub-actions
- MTP compatibility flag per sub-action
- KB entries for model-specific vendor kernel behavior

## Heuristic Update

- Kept sub-action: boost similar sub-actions for same model class by 1.3×
- Reverted sub-action: reduce score for that sub-action by 0.5× for this model class
- MTP-incompatible result: add `mtp_incompatible` tag to KB entry, reduce score to 0
  when MTP is active

## Failure Handling

- Server crash after config change: revert all env vars, restart with known-good config
- Import error (aiter version too old): log version requirement, skip sub-action
- Kernel not found (variant doesn't exist in this aiter version): skip, log
