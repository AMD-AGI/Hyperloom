# Action: Baseline Training Run (Full Convergence)

## Overview

Establishes the real, measured TTT via a Tier 4 full convergence run. All subsequent
optimizations compare against this baseline. The baseline MUST be Tier 4 (not Tier 1/2/3).
Reference TTT: ~206 minutes — this run re-verifies it.

## Inputs

- `$MLPERF_DIR`, `$CONFIG_SH`, `$EXP`
- `$RESULT_DIR` for storing outputs
- `$SKILL_ROOT` for scripts

## Procedure

### Step 1: Source common helpers

```bash
source "$SKILL_ROOT/scripts/common.sh"
```

### Step 2: Run baseline training (Tier 4 — full convergence)

This runs with the current config, no overrides, no timeout. Training continues
until either the target eval_loss of 3.34 is reached (`run_stop status=success`)
or all iterations are exhausted (`run_stop status=aborted`).

```bash
run_mlperf_trial "baseline" 4
```

**Do NOT interrupt this run.** It must complete naturally to establish a valid baseline.

Tier 4 runs with:
- Original `PRIMUS_TRAIN_ITERS` (full iteration count from config)
- Original `PRIMUS_EVAL_INTERVAL` (standard eval cadence)
- `MLLOG_TRAIN_LOSS_LOG_FREQ=32` (original)
- Full YAML verbosity (no quiet config)
- **No timeout**

### Step 3: Extract baseline TTT (primary metric)

```bash
TTT_INFO=$(extract_time_to_train "$RESULT_DIR/attempt_baseline_raw.log")
baseline_ttt_seconds=$(echo "$TTT_INFO" | cut -f1)
baseline_ttt_status=$(echo "$TTT_INFO" | cut -f2)
baseline_ttt_minutes=$(python3 -c "print(f'{float(\"$baseline_ttt_seconds\") / 60:.1f}')")

echo "Baseline TTT: ${baseline_ttt_minutes} min (status: ${baseline_ttt_status})"
echo "Reference TTT: 206 min"
```

**If `baseline_ttt_status` is `aborted`:** The baseline did not converge. This is
a critical issue — document the final eval_loss and investigate before proceeding.

### Step 4: Parse TRIAL_RESULT for ms/iter

```bash
eval "$(parse_trial_result "$(grep TRIAL_RESULT $RESULT_DIR/attempt_baseline.log)")"
baseline_ms_per_iter="$TRIAL_MS_PER_ITER"
baseline_gbs="$TRIAL_GBS"
baseline_last_loss="$TRIAL_LAST_LOSS"
baseline_run_status="$TRIAL_RUN_STATUS"
```

### Step 5: Verify GBS from raw log

```bash
verify_gbs "$RESULT_DIR/attempt_baseline_raw.log" "$PRIMUS_GLOBAL_BATCH_SIZE"
```

### Step 6: Extract final eval_loss

```bash
python3 -c "
import json
evals = []
with open('$RESULT_DIR/attempt_baseline_raw.log') as f:
    for line in f:
        if not line.startswith(':::MLLOG'):
            continue
        data = json.loads(line.replace(':::MLLOG ', ''))
        if data['key'] == 'eval_accuracy':
            evals.append(data['value'])
if evals:
    print(f'Final eval_loss: {evals[-1]:.4f} (target: 3.34)')
    print(f'Total evals: {len(evals)}')
"
```

### Step 7: Initialize results log

```
cat > "$RESULT_DIR/results.tsv" <<EOF
attempt	ms_per_iter	ttt_seconds	ttt_minutes	status	description
0	${baseline_ms_per_iter}	${baseline_ttt_seconds}	${baseline_ttt_minutes}	baseline	Baseline full run (${GPUS_PER_NODE} GPU, GBS=${baseline_gbs}, FP8=hybrid, TTT=${baseline_ttt_minutes}min)
EOF
```

## Outputs

- `baseline_ttt_seconds`: actual time-to-train in seconds (primary baseline metric)
- `baseline_ttt_minutes`: TTT in minutes for readability
- `baseline_ms_per_iter`: average ms/iter (from TRIAL_RESULT)
- `baseline_gbs`: verified global batch size
- `baseline_run_status`: `success` (converged) or `aborted`
- `$RESULT_DIR/attempt_baseline_raw.log`: full raw training log (complete Tier 4 run)
- `$RESULT_DIR/attempt_baseline.log`: filtered training log
- `$RESULT_DIR/results.tsv`: initialized results tracking with TTT

## Heuristic Update

N/A — baseline establishes the reference point for all subsequent actions.

## Failure Handling

- If `run_stop status=aborted` (did not converge): STOP. Investigate config.
  Do not proceed with optimization until the baseline can converge.
- If `TRIAL_RESULT` shows `status=nan`: FP8 instability, try reducing LR or using BF16
- If training crashes: check for port conflicts (increment `MASTER_PORT`), OOM, missing deps
- If TTT differs significantly from reference 206 min: document the difference and investigate
