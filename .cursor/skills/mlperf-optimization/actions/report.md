# Action: Generate Optimization Report

## Overview

Generates the final MLPerf optimization report. This action runs a **complete Tier 3 trial
(full convergence run)** — the training must NOT be interrupted. It runs until the model
reaches the target eval loss of 3.34 (`run_stop status=success`) or exhausts all iterations
(`run_stop status=aborted`). The resulting **time-to-train (TTT)** is the primary metric
that demonstrates the value of all optimizations.

## Inputs
- `$RESULT_DIR` with `results.tsv`, traces, logs
- All state from the optimization run (kept optimizations applied)

## Procedure

### Step 0: REVIEW CHECKPOINT RC-7 (pre-flight)

Before running the final verification, confirm:
- All kept optimizations are applied to the config
- YAML and env overrides are consistent
- Previous Tier 1/2 trials showed the expected cumulative gain
- No stale `.bak` files or leftover env overrides from earlier trials

### Step 1: Run final optimized config (Tier 3 — full run to convergence)

```bash
source "$SKILL_ROOT/scripts/common.sh"
run_mlperf_trial "final" 3
```

**CRITICAL: Do NOT interrupt this run.**

Tier 3 runs with:
- Original `PRIMUS_TRAIN_ITERS` (full iteration count from config)
- Original `PRIMUS_EVAL_INTERVAL` (standard eval cadence)
- `MLLOG_TRAIN_LOSS_LOG_FREQ=32` (original)
- Full YAML verbosity (no quiet config)
- **No timeout** — the run completes naturally

The training will:
1. Train until the target eval loss of 3.34 is reached, at which point `primus_mllog`
   emits `run_stop` with `status=success` and exits automatically.
2. OR: exhaust all `PRIMUS_TRAIN_ITERS` without reaching the target, emitting `run_stop`
   with `status=aborted`.

Wait for the process to fully exit before proceeding.

### Step 2: Extract final metrics from raw log

```bash
source "$SKILL_ROOT/scripts/common.sh"

# Time-to-train (the primary result)
TTT_INFO=$(extract_time_to_train "$RESULT_DIR/attempt_final_raw.log")
TTT_SECONDS=$(echo "$TTT_INFO" | cut -f1)
TTT_STATUS=$(echo "$TTT_INFO" | cut -f2)

# ms/iter (steady-state, warmup excluded)
FINAL_MS=$(extract_ms_per_iter "$RESULT_DIR/attempt_final_raw.log" 10 50)

# TRIAL_RESULT from filtered log
eval "$(parse_trial_result "$(grep TRIAL_RESULT $RESULT_DIR/attempt_final.log)")"

# Extract all eval_accuracy values to find the final one
python3 -c "
import json
evals = []
with open('$RESULT_DIR/attempt_final_raw.log') as f:
    for line in f:
        if not line.startswith(':::MLLOG'):
            continue
        data = json.loads(line.replace(':::MLLOG ', ''))
        if data['key'] == 'eval_accuracy':
            evals.append(data['value'])
if evals:
    print(f'Final eval_loss: {evals[-1]:.4f} (target: 3.34)')
    print(f'Total evals: {len(evals)}')
    if evals[-1] <= 3.34:
        print('TARGET REACHED')
    else:
        print(f'Target NOT reached. Gap: {evals[-1] - 3.34:.4f}')
else:
    print('No eval_accuracy events found')
"
```

### Step 3: REVIEW CHECKPOINT RC-7 (post-run)

Verify:
- `TTT_STATUS` is `success` (target was reached)
  - If `success`: TTT is the definitive result — record it
  - If `aborted`: analyze final eval_loss, estimate how many more iterations were needed
- MLLOG compliance: `run_start` and `run_stop` events both present
- RESULT line present in filtered log
- No NaN/Inf anomalies in the run

### Step 4: Write optimization report

Write to `$RESULT_DIR/optimization_report.md`:

```markdown
# GPT-OSS-20B MLPerf Training Optimization Report — MI355X 8-GPU

**Date:** YYYY-MM-DD
**Platform:** 8× AMD MI355X
**Benchmark:** gpt-oss-20b (MLPerf Training 5.1.0)
**Quality target:** validation log perplexity = 3.34

---

## Executive Summary

| Metric | Baseline | Optimized | Delta |
|---|---|---|---|
| **ms/iter** | X | Y | **-Z ms (-W%)** |
| **Time-to-target (TTT)** | A s | B s | **-C s (-D%)** |
| **Final eval loss** | — | V | target: 3.34 |
| **Convergence status** | — | success/aborted | |
| **GBS** | 32 | G | |
| Kept optimizations | — | M of N | |

**Final config:**
- GBS: ...
- LR: ...
- Fusion flags: ...
- Parallelism: ...
- Environment: ...

---

## Convergence Result

**Run status:** success / aborted
**Time-to-target:** X seconds (X minutes)
**Total iterations:** N
**Final eval loss:** V (target: 3.34)
**Eval loss trajectory:** [list of eval_accuracy values from MLLOG]

---

## All Attempts

| # | ms/iter | Δ vs baseline | Status | Description |
|---|---|---|---|---|
| 0 | X | — | baseline | ... |
| 1 | Y | -Z% | keep | ... |

## What Worked
- ...

## What Didn't Work
- ...

## Kernel Profile Comparison
(baseline vs final top-20 kernels, if profiled)

## Recommendations for Production
- ...

## Reproducibility

**Baseline command:**
```bash
cd /root/Hyperloom-plus-mlperf/training_optimization/mlperf
source config_MI355X_1x8x1_fp8.sh
bash setup_container_symlinks.sh
source config_MI355X_1x8x1_fp8.sh && bash run_and_time.sh
```

**Optimized command:**
```bash
cd /root/Hyperloom-plus-mlperf/training_optimization/mlperf
source config_MI355X_1x8x1_fp8.sh
<env var overrides>
bash setup_container_symlinks.sh
source config_MI355X_1x8x1_fp8.sh && bash run_and_time.sh
```
```

### Step 5: Ingest key findings into KB

```bash
python3 $SKILL_ROOT/kb/kb_ingest.py \
    --category "final_result" \
    --model "GPT-OSS-20B" \
    --action "Full Tier 3 convergence run with all optimizations" \
    --lesson "TTT=${TTT_SECONDS}s, status=${TTT_STATUS}, final_eval_loss=${FINAL_EVAL_LOSS}, ms_per_iter=${FINAL_MS}" \
    --tags "mlperf,training,MI355X,gpt-oss-20b,tier3,convergence" \
    --gain "$CUMULATIVE_GAIN_PCT" \
    --status "$TTT_STATUS"
```

## Outputs
- `$RESULT_DIR/optimization_report.md`: comprehensive report with TTT result
- `$RESULT_DIR/attempt_final_raw.log`: full raw training log (Tier 3, complete run)
- `$RESULT_DIR/attempt_final.log`: filtered training log
- KB entries for all key findings including convergence result

## Failure Handling
- If `run_stop status=aborted` (target not reached):
  - Report the final eval_loss and gap to target
  - Estimate iterations needed based on loss curve slope
  - Check if any optimization degraded convergence (compare Tier 2 eval trends)
  - Include analysis in report under "Why Target Was Not Reached"
- If final run crashes: write report with best-so-far config and crash analysis
- If final run hangs (no progress for >30 min): check GPU utilization, but do NOT kill —
  wait for the process to recover or fail naturally
