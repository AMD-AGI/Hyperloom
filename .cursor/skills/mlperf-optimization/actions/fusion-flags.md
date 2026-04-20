# Action: Fusion Flag Exploration

## Overview

Fusion flags replace generic PyTorch kernels with fused, hardware-optimized alternatives.
This is the **highest-impact action for MoE models** — each flag is a single config override,
zero code changes, zero crash risk.

## Inputs

- Baseline ms/iter and kept_overrides from prior attempts
- Model class from classify step
- Profile data (which kernels dominate)
- Current YAML config state

## KB Query

```
python3 $SKILL_ROOT/kb/kb_query.py "GPT-OSS-20B fusion flags MoE" --top-k 5 --compact
```

## Pre-Check: Already-Enabled Flags

GPT-OSS-20B config already enables several fusion flags by default. Check before testing:

```python
already_enabled = {
    "moe_permute_fusion": True,         # Already in config
    "gradient_accumulation_fusion": True, # Already in config
    "apply_rope_fusion": True,           # Already in config
    "moe_grouped_gemm": True,            # Already in config
    "moe_router_fusion": True,           # Already in config
    "cross_entropy_loss_fusion": True,    # Already in config
}
```

## Fusion Flag Matrix

Test flags NOT already enabled:

### Tier 1: High-confidence flags (try first)

| Flag | Expected Impact | Notes |
|------|----------------|-------|
| `moe_use_fused_router_with_aux_score=true` | ~0–0.5% | Currently `false` in config. Fused TopK router |
| `use_turbo_grouped_mlp=true` | -1% to +1% | Currently `false`. Fused SwiGLU for MoE |
| `use_turbo_attention=true` | +0–1% | Currently `false`. PrimusTurbo attention |
| `moe_shared_expert_overlap=true` | +0–1% | Overlap shared expert with routing |

### Tier 2: DeepEP and sync-free MoE

| Flag | Expected Impact | Notes |
|------|----------------|-------|
| `moe_enable_deepep=true` + `use_turbo_deepep=true` | +2–5% | Enable DeepEP for expert comm |
| `turbo_sync_free_moe_stage=2` | +1–3% | Sync-free MoE pipeline (requires legacy grouped gemm) |
| `turbo_sync_free_moe_stage=3` | +2–4% | Full sync-free (more memory) |

### Tier 3: TE/FP8 specific

| Flag | Expected Impact | Notes |
|------|----------------|-------|
| `NVTE_USE_CAST_TRANSPOSE_TRITON=1` | **+0.2% (confirmed)** | Triton-based FP8 cast+transpose. Known prior result — apply early. |
| `NVTE_ROCM_ENABLE_MXFP8=1` | Variable | MX-FP8 (newer format, not all kernels support) |

### Flags to AVOID

| Flag | Why |
|------|-----|
| Changing `window_size` or `window_attn_skip_freq` | Affects model quality |
| `moe_use_legacy_grouped_gemm=false` | Required for current EP=1 config |

## Procedure

### Step 1: Test each flag individually (Tier 1)

For each untested flag, modify the YAML config and run a Tier 1 trial:

```bash
source "$SKILL_ROOT/scripts/common.sh"

# Edit YAML to change the flag (e.g., use_turbo_attention: true)
# Then run:
run_mlperf_trial "fusion_FLAG_NAME" 1
```

Parse the `TRIAL_RESULT` line to get ms/iter:
```bash
eval "$(parse_trial_result "$(grep TRIAL_RESULT $RESULT_DIR/attempt_fusion_FLAG_NAME.log)")"
gain=$(compute_gain_pct "$baseline_ms_per_iter" "$TRIAL_MS_PER_ITER")
```

### Step 1.5: Validate winners with Tier 2

For any flag that improved ms/iter by >1%, run a convergence validation:

```bash
run_mlperf_trial "fusion_FLAG_NAME_validate" 2 500
```

Check that `TRIAL_STATUS != "nan"` and loss trajectory is stable.

### Step 2: Measure and decide

- **Improved (gain > 1% confirmed by Tier 2):** KEEP — update YAML config permanently
- **Marginal (0–1%):** KEEP tentatively, verify in combination test
- **Same or worse:** DISCARD — revert YAML change
- **Crashed or NaN:** Log the error, mark as `crash`/`nan`, revert YAML

### Step 3: Combine winners (Tier 2)

After testing all flags individually, test all KEPT flags together with Tier 2:

```bash
# Apply all winning flags to YAML
run_mlperf_trial "fusion_combined" 2 500
```

If combined result is better than individual best, the combination is the new baseline.

## Outputs
- `winning_flags`: list of flags that improved ms/iter
- Combined gain percentage
- Updated YAML config with winning flags
- KB entries for each tested flag

## Heuristic Update

- Individual flag gain > 1%: boost remaining untested flags by 1.5x
- All flags neutral/negative: reduce fusion-flags score by 0.7x
- After 2+ wins: push combined_fusion_test (Score Update Rule #3)
- After all flags tested: push re-profile (Score Update Rule #4)

## Failure Handling

- If a flag crashes: log the error, mark as `crash`, revert YAML
- If a flag causes loss divergence: mark as `convergence_fail`, revert
- If all flags fail: move to comm-tuning or kernel-opt
