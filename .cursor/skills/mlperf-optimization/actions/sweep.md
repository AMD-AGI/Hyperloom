# Action: Parameter Sweep

## Overview

After the DFS optimization loop completes, run a systematic sweep over key parameters
to find the optimal operating point for time-to-target.

## Inputs
- Final kept_overrides and kept_patches from DFS loop
- Current best config

## Sweep Dimensions

### Primary: GBS × LR

| GBS | LR | MBS | GA | Notes |
|-----|----|-----|-----|-------|
| 16 | 2.0e-4 | 2 | 1 | Conservative |
| 32 | 4.0e-4 | 2 | 2 | Current default |
| 32 | 4.0e-4 | 4 | 1 | Larger MBS, less GA |
| 64 | 5.6e-4 | 2 | 4 | Aggressive |
| 64 | 5.6e-4 | 4 | 2 | Aggressive + large MBS |

### Secondary: Eval Interval

| eval_samples_interval | Overhead per eval | Notes |
|----------------------|-------------------|-------|
| 6144 | ~30s | More frequent |
| 12288 | ~30s | Default |
| 24576 | ~30s | Less frequent |

## Procedure

### Step 1: Build sweep configs

```python
sweep_configs = []
for gbs, lr in [(16, 2e-4), (32, 4e-4), (64, 5.6e-4)]:
    for mbs in [2, 4]:
        ga = gbs // (mbs * dp)
        if ga < 1:
            continue
        sweep_configs.append({
            "gbs": gbs, "lr": lr, "mbs": mbs, "ga": ga,
            "label": f"gbs{gbs}_lr{lr}_mbs{mbs}"
        })
```

### Step 2: Run each config (Tier 2 trial)

For each config, run a Tier 2 convergence trial:

```bash
source "$SKILL_ROOT/scripts/common.sh"
run_mlperf_trial "sweep_${label}" 2 500 \
    "PRIMUS_GLOBAL_BATCH_SIZE=$gbs PRIMUS_LR=$lr PRIMUS_MICRO_BATCH_SIZE=$mbs"

eval "$(parse_trial_result "$(grep TRIAL_RESULT $RESULT_DIR/attempt_sweep_${label}.log)")"
```

### Step 3: Estimate time-to-target

```python
# For each config:
# 1. ms/iter from MLLOG timestamps
# 2. Loss slope (loss reduction per sample)
# 3. Projected iters to target: extrapolate from loss curve
# 4. Total projected time = (iters_to_target × ms_per_iter / 1000) + (num_evals × 30s)
```

### Step 4: Select optimal config

Pick the config with minimum projected time-to-target that also shows stable convergence.

## Outputs
- `$RESULT_DIR/sweep_results.tsv`: full sweep data
- Optimal GBS, LR, MBS, eval_interval combination
- Projected time-to-target for winning config

## Failure Handling
- OOM on large MBS: record as `oom`, skip
- Divergence on high LR: record as `diverged`, skip
