# Action: Parameter Sweep (MBS / Eval Interval Refinement)

## Overview

After the DFS optimization loop and config selection complete, run a sweep over
Micro Batch Size (MBS) and eval interval to find the optimal operating point within
the winning GBS/LR/parallelism configuration.

**Note:** GBS, LR, and TP/EP/DP selection is handled by
[`actions/config-selection.md`](config-selection.md). This sweep explores MBS and
eval interval variations that do not change the convergence trajectory — only the
per-iteration cost and eval overhead.

## Inputs

- Winning config from config-selection (GBS, LR, EP, TP, DP)
- Final kept_overrides and kept_patches from DFS loop
- Baseline ms/iter with winning config

## KB Query

```
python3 $SKILL_ROOT/kb/kb_query.py "GPT-OSS-20B MBS sweep" --top-k 5 --compact
```

## Sweep Dimensions

### Primary: MBS × GA (maintaining winning GBS)

| MBS | GA (for GBS=32, DP=8) | Notes |
|-----|------------------------|-------|
| 1 | 4 | Smaller MBS, more GA — lower memory, more overhead |
| 2 | 2 | Current default |
| 4 | 1 | Larger MBS, no GA — higher memory, less overhead |

GA is recomputed as `GBS / (MBS × DP)`. Only valid configs (integer GA ≥ 1) are tested.

### Secondary: Eval Interval

| eval_samples_interval | Overhead per eval | Notes |
|----------------------|-------------------|-------|
| 6144 | ~30s | More frequent, catches target sooner |
| 12288 | ~30s | Default |
| 24576 | ~30s | Less frequent, saves wall time |

## Procedure

### Step 1: Build sweep configs

```python
winning_gbs = state["global_batch_size"]
winning_lr = state["kept_env_vars"].get("PRIMUS_LR", "4.0e-4")
dp = state["num_gpus"] // (state["tp"] * 1)  # PP=1

sweep_configs = []
for mbs in [1, 2, 4]:
    ga = winning_gbs // (mbs * dp)
    if ga < 1 or ga * mbs * dp != winning_gbs:
        continue
    sweep_configs.append({
        "gbs": winning_gbs, "lr": winning_lr, "mbs": mbs, "ga": ga,
        "label": f"mbs{mbs}_ga{ga}"
    })
```

### Step 2: Run each config (Tier 2 trial)

```bash
source "$SKILL_ROOT/scripts/common.sh"
run_mlperf_trial "sweep_${label}" 2 500 \
    "PRIMUS_MICRO_BATCH_SIZE=$mbs"

eval "$(parse_trial_result "$(grep TRIAL_RESULT $RESULT_DIR/attempt_sweep_${label}.log)")"
```

### Step 3: Estimate time-to-target

```python
# For each config:
# 1. ms/iter from TRIAL_RESULT
# 2. Eval overhead: num_evals × 30s (estimate from eval_interval)
# 3. Total projected time = (projected_iters × ms_per_iter / 1000) + eval_overhead
```

### Step 4: Select optimal config

Pick the MBS/eval_interval with minimum projected time-to-target. Since GBS/LR
are fixed, convergence trajectory is the same — only wall-time cost differs.

## Outputs

- `$RESULT_DIR/sweep_results.tsv`: MBS/eval sweep data
- Optimal MBS and eval_interval for the winning config
- Updated ms/iter after MBS optimization

## Heuristic Update

N/A — sweep is a post-DFS refinement step.

## Failure Handling

- OOM on larger MBS: record as `oom`, skip that MBS
- If no MBS improves over default: keep MBS=2
