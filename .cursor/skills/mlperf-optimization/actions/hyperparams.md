# Action: Hyperparameter Tuning (MLPerf-specific)

## Overview

Unlike standard training optimization (where GBS is fixed), MLPerf allows tuning
hyperparameters that affect convergence speed. The goal is to minimize
time-to-target (wall seconds to reach validation loss 3.34).

This is a **high-impact, high-risk** action — wrong hyperparameters can prevent
convergence entirely.

## Inputs
- Baseline ms/iter, GBS, LR, warmup
- Loss trajectory from baseline run
- Eval loss convergence rate

## Hyperparameter Search Space

### Global Batch Size (GBS)

| GBS | GA Steps (MBS=2, DP=8) | Iters to 12288 samples | Notes |
|-----|-------------------------|------------------------|-------|
| 16 | 1 | 768 | Original MLPerf default. Fewer samples/iter but more iters |
| 32 | 2 | 384 | Current config. Good balance |
| 64 | 4 | 192 | Fewer iters, each slower. Needs LR adjustment |
| 128 | 8 | 96 | Very few iters. LR scaling critical |

**GBS scaling rule:** When increasing GBS by factor k, scale LR by sqrt(k) and
adjust warmup proportionally.

### Learning Rate (LR)

| LR | GBS | Notes |
|----|-----|-------|
| 4.0e-4 | 32 | Current config |
| 5.6e-4 | 64 | sqrt(2) × 4e-4 |
| 8.0e-4 | 128 | 2 × 4e-4 |
| 2.0e-4 | 16 | 0.5 × 4e-4 |

### Warmup Iterations

| Warmup | GBS | Notes |
|--------|-----|-------|
| 128 | 32 | Current (default) |
| 64 | 64 | Halve with doubled GBS |
| 32 | 128 | Quarter with 4× GBS |
| 256 | 16 | Double with halved GBS |

### Eval Interval Optimization

| eval_samples_interval | eval_interval (GBS=32) | Overhead | Notes |
|----------------------|------------------------|----------|-------|
| 12288 | 384 | ~30s every 384 iters | Current default |
| 24576 | 768 | ~30s every 768 iters | Less overhead, risk overshooting |
| 6144 | 192 | ~30s every 192 iters | More frequent, catches target sooner |

## Procedure

### Step 1: Test GBS scaling

For each candidate GBS, run a Tier 2 convergence trial (500 iters with eval enabled):

```bash
source "$SKILL_ROOT/scripts/common.sh"

# GBS=64 example
run_mlperf_trial "gbs64" 2 500 \
    "PRIMUS_GLOBAL_BATCH_SIZE=64 PRIMUS_LR=5.6e-4 PRIMUS_MIN_LR=5.6e-5 PRIMUS_LR_WARMUP_ITERS=64"
```

If the `TRIAL_RESULT` shows `status=nan`, skip this GBS config immediately.

### Step 2: Compare convergence rates

From each trial's `TRIAL_RESULT` and raw log:

```bash
eval "$(parse_trial_result "$(grep TRIAL_RESULT $RESULT_DIR/attempt_gbs64.log)")"
ms_iter="$TRIAL_MS_PER_ITER"
gbs="$TRIAL_GBS"
samples_per_sec=$(python3 -c "print(f'{$gbs / ($ms_iter / 1000):.1f}')")

# Extract losses from raw log for convergence slope
extract_losses "$RESULT_DIR/attempt_gbs64_raw.log"
```

### Step 3: GBS decision matrix

Score each config by projected time-to-target:
- `ms/iter × projected_iters_to_target = total_time`
- Loss convergence rate (slope of loss vs samples)
- Eval overhead: `num_evals × 30s`
- Stability: `TRIAL_STATUS == "ok"` and no loss spikes

### Step 4: Validate winning config (Tier 2 extended)

Run the winning config with 200+ iterations to confirm convergence trend:

```bash
run_mlperf_trial "gbs_winner_extended" 2 800 "PRIMUS_GLOBAL_BATCH_SIZE=..."
```

## Outputs
- Optimal GBS, LR, warmup, eval_interval combination
- Projected time-to-target for each tested config
- Updated environment variables

## Failure Handling
- If larger GBS causes NaN/divergence: LR too high, try lower scaling factor
- If convergence slows dramatically: GBS too large for this model
- If OOM with larger MBS: use more GA steps instead
- Always revert to known-good config before trying next candidate
