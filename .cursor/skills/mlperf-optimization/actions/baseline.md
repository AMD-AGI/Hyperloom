# Action: Baseline Training Run

## Inputs
- `$MLPERF_DIR`, `$CONFIG_SH`, `$EXP`
- `$RESULT_DIR` for storing outputs
- `$SKILL_ROOT` for scripts

## Procedure

### Step 1: Source common helpers

```bash
source "$SKILL_ROOT/scripts/common.sh"
```

### Step 2: Run baseline training (Tier 1 — 100 iters)

Uses `run_mlperf_trial` which automatically:
- Sets `MLLOG_TRAIN_LOSS_LOG_FREQ=1` (every iteration logged)
- Applies quiet YAML (suppresses Megatron DEBUG/INFO noise)
- Filters output through `trial_monitor.py`
- Preserves raw log at `$RESULT_DIR/attempt_baseline_raw.log`

```bash
run_mlperf_trial "baseline" 1
```

### Step 3: Parse TRIAL_RESULT

```bash
eval "$(parse_trial_result "$(grep TRIAL_RESULT $RESULT_DIR/attempt_baseline.log)")"
baseline_ms_per_iter="$TRIAL_MS_PER_ITER"
baseline_gbs="$TRIAL_GBS"
baseline_last_loss="$TRIAL_LAST_LOSS"
```

### Step 4: Sanity check

```python
# ms/iter should be in expected range for GPT-OSS-20B on 8× MI355X
if baseline_ms_per_iter < 500 or baseline_ms_per_iter > 10000:
    print(f"WARNING: ms/iter={baseline_ms_per_iter} is outside expected range")
```

### Step 5: Verify GBS from raw log

```bash
verify_gbs "$RESULT_DIR/attempt_baseline_raw.log" "$PRIMUS_GLOBAL_BATCH_SIZE"
```

### Step 6: Extract loss trajectory from raw log

```bash
echo "Loss trajectory:"
extract_losses "$RESULT_DIR/attempt_baseline_raw.log"
```

### Step 7: Initialize results log

```
cat > "$RESULT_DIR/results.tsv" <<EOF
attempt	ms_per_iter	speedup_pct	status	description
0	${baseline_ms_per_iter}	0.0	baseline	Baseline (${GPUS_PER_NODE} GPU, GBS=${baseline_gbs}, FP8=hybrid)
EOF
```

## Outputs
- `baseline_ms_per_iter`: average ms/iter (warmup-excluded)
- `baseline_gbs`: verified global batch size
- `baseline_last_loss`: last training loss from the trial
- `$RESULT_DIR/attempt_baseline_raw.log`: full raw training log
- `$RESULT_DIR/attempt_baseline.log`: filtered training log
- `$RESULT_DIR/results.tsv`: initialized results tracking

## Failure Handling
- If `TRIAL_RESULT` shows `status=nan`: FP8 instability, try reducing LR or using BF16
- If `TRIAL_RESULT` shows `status=no_data`: check that primus_mllog is installed
- If training crashes: check for port conflicts (increment `MASTER_PORT`), OOM, missing deps
- If ms/iter is outside expected range: check GPU utilization, data loading
