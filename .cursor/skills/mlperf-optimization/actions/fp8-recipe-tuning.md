# Action: FP8 Recipe Tuning

## Overview

Explores alternative FP8 configurations and TE knobs that can improve throughput or
convergence. This is a convergence-affecting action: all FP8 changes require Tier 2
minimum, and NaN in the first 200 iterations triggers immediate revert.

## Inputs

- Baseline ms/iter and TTT
- Current FP8 config from `config_MI355X_1x8x1_fp8.sh`
- Profile data (fp8_ops category percentage from profile step)

## KB Query

```
python3 $SKILL_ROOT/kb/kb_query.py "GPT-OSS-20B FP8 recipe tuning" --top-k 5 --compact
```

## FP8 Knob Matrix

### Tier 1: Confirmed / Low-Risk (apply first)

| Knob | Current | Target | Risk | Expected Impact |
|------|---------|--------|------|-----------------|
| `NVTE_USE_CAST_TRANSPOSE_TRITON` | **0** | **1** | None | **+0.2% (confirmed)** |

**KB reference:** Entry `12abcd22` confirms +0.2% gain with zero convergence risk.

### Tier 2: Medium-Risk (test individually)

| Knob | Current | Candidates | Risk | Notes |
|------|---------|-----------|------|-------|
| `NVTE_CK_IS_V3_ATOMIC_FP32` | 0 | 1 | Medium | FP32 atomics in CK v3 backward — may improve numerical precision at small cost |
| `fp8_amax_history_len` | default | 4, 16, 64 | Low | Scaling factor history window — shorter = more responsive, longer = more stable |
| `fp8_amax_compute_algo` | default | max, most_recent | Low | How amax is computed for FP8 scaling |

### Tier 3: High-Risk (experimental)

| Knob | Current | Target | Risk | Notes |
|------|---------|--------|------|-------|
| `NVTE_ROCM_ENABLE_MXFP8` | 0 | 1 | High | MX-FP8 (microscaling format) — newer, not all kernels support it |

## Procedure

### Step 1: Apply confirmed improvement (`NVTE_USE_CAST_TRANSPOSE_TRITON=1`)

In `config_MI355X_1x8x1_fp8.sh` line 77, change value from `0` to `1`.

Verify with Tier 1:

```bash
source "$SKILL_ROOT/scripts/common.sh"
run_mlperf_trial "fp8_triton_cast" 1 "" "NVTE_USE_CAST_TRANSPOSE_TRITON=1"
eval "$(parse_trial_result "$(grep TRIAL_RESULT $RESULT_DIR/attempt_fp8_triton_cast.log)")"
gain=$(compute_gain_pct "$baseline_ms_per_iter" "$TRIAL_MS_PER_ITER")
echo "NVTE_USE_CAST_TRANSPOSE_TRITON=1: gain=${gain}%"
```

### Step 2: Test each Tier 2 knob (Tier 2, 500 iters)

For each knob, run with eval enabled to catch convergence issues:

```bash
run_mlperf_trial "fp8_atomic_fp32" 2 500 "NVTE_CK_IS_V3_ATOMIC_FP32=1"

# amax history length variants
run_mlperf_trial "fp8_amax_hist4" 2 500 "NVTE_FP8_AMAX_HISTORY_LEN=4"
run_mlperf_trial "fp8_amax_hist64" 2 500 "NVTE_FP8_AMAX_HISTORY_LEN=64"
```

For each trial:
1. Check `TRIAL_STATUS` — if `nan`, discard immediately
2. Compare ms/iter against baseline — is there a throughput change?
3. Compare eval_loss trajectory — is convergence degraded?
4. Extract loss-vs-samples curve:
   ```bash
   extract_eval_trajectory "$RESULT_DIR/attempt_fp8_KNOB_raw.log"
   ```

Decision:
- NaN → DISCARD, revert, log to KB as unsafe
- `loss_efficiency < baseline × 0.85` → DISCARD
- ms/iter improved + convergence stable → KEEP
- Ambiguous (±15%) → escalate to Tier 2.5, KEEP if `ttt_gain_pct > 0%`

### Step 3: MX-FP8 exploration (Tier 3 knob)

Only proceed if all Tier 2 knobs are tested and MX-FP8 is worth exploring:

```bash
run_mlperf_trial "fp8_mxfp8_check" 1 "" "NVTE_ROCM_ENABLE_MXFP8=1"
eval "$(parse_trial_result "$(grep TRIAL_RESULT $RESULT_DIR/attempt_fp8_mxfp8_check.log)")"

if [ "$TRIAL_STATUS" = "nan" ] || [ "$TRIAL_STATUS" = "no_data" ]; then
    echo "MX-FP8 failed NaN check — skipping"
else
    # Extended convergence check (Tier 2L)
    run_mlperf_trial "fp8_mxfp8_conv" 2L "" "NVTE_ROCM_ENABLE_MXFP8=1"
fi
```

### Step 4: Combined FP8 tuning validation

Apply all winning FP8 knobs together:

```bash
FP8_COMBINED="NVTE_USE_CAST_TRANSPOSE_TRITON=1"
# Append other winners as discovered in Steps 2-3

run_mlperf_trial "fp8_combined" 2L "" "$FP8_COMBINED"
```

Project TTT from the combined trial:

```bash
projected=$(project_ttt "$RESULT_DIR/attempt_fp8_combined_raw.log" "$GBS")
echo "FP8 combined projected TTT: $projected"
```

## Outputs

- List of winning FP8 knobs with individual gain percentages
- Combined gain from all FP8 optimizations
- Updated environment variables in `state.kept_env_vars`
- Updated `state.fp8_knobs_tested`
- KB entries for each tested knob (pass/fail/gain)

## Heuristic Update

- FP8 knob succeeds: boost remaining untested FP8 knobs by 1.3x (Score Update Rule #8)
- All knobs fail: keep confirmed NVTE_USE_CAST_TRANSPOSE_TRITON=1, reduce fp8-recipe score

## Failure Handling

- NaN in first 200 iters → REVERT immediately, mark as unsafe
- Loss divergence (eval_loss increasing over 500 iters) → REVERT, mark as convergence risk
- If all Tier 2/3 knobs fail: keep only the confirmed `NVTE_USE_CAST_TRANSPOSE_TRITON=1`
- MX-FP8 kernel not supported error → skip, log to KB
