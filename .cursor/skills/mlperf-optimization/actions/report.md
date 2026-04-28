# Action: Generate Optimization Report

## Overview

This action produces the final MLPerf optimization report by running a **complete Tier 4 trial** (full convergence) and treating **time-to-train (TTT)** as the primary value metric. Training runs until validation reaches target eval loss 3.34 (`run_stop status=success`) or iterations are exhausted (`status=aborted`). After that run, a short profiling pass plus TraceLens CLI analysis captures the before/after kernel bottleneck picture against the baseline trace.

## Inputs

- `$RESULT_DIR` with `results.tsv`, traces, logs; `$RESULT_DIR/baseline_trace.json` from the initial profile step; full optimization state (kept optimizations applied).

## Procedure

### Step 1: Run final optimized config (Tier 4 — full run to convergence)

Before launching, confirm kept optimizations and YAML/env overrides are consistent, prior Tier 1/2 trials show the expected cumulative gain, and there are no stale `.bak` files or leftover env overrides from earlier trials.

```bash
source "$SKILL_ROOT/scripts/common.sh"
run_mlperf_trial "final" 4
```

Tier 4 uses the config’s full `PRIMUS_TRAIN_ITERS`, normal `PRIMUS_EVAL_INTERVAL`, `MLLOG_TRAIN_LOSS_LOG_FREQ=32`, full YAML verbosity (no quiet config), and no timeout—the process exits when `run_stop` fires (`success` if target loss is reached, `aborted` if iterations run out). Wait for the process to fully exit before proceeding.

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

### Step 3: Verify results and logs

- `TTT_STATUS`: if `success`, record TTT as definitive; if `aborted`, analyze final eval_loss and estimate iterations still needed.
- MLLOG: `run_start` and `run_stop` present; RESULT line in filtered log; no NaN/Inf anomalies.

### Step 4: Re-profile optimized config

Short profiling pass with kept optimizations to capture a final kernel trace vs. baseline.

```python
import yaml, os

config_path = EXP
TB_TRACE_DIR = os.path.join(RESULT_DIR, "tb_traces")
os.makedirs(TB_TRACE_DIR, exist_ok=True)

with open(config_path) as f:
    config = yaml.safe_load(f)
overrides = config["modules"]["pre_trainer"]["overrides"]
overrides["profile"] = True
overrides["use_pytorch_profiler"] = True
overrides["profile_step_start"] = 5
overrides["profile_step_end"] = 8
overrides["tensorboard_dir"] = TB_TRACE_DIR
overrides["torch_profiler_use_gzip"] = True
with open(config_path, "w") as f:
    yaml.dump(config, f, default_flow_style=False)
```

```bash
source "$SKILL_ROOT/scripts/common.sh"
run_mlperf_trial "final_profile" 1 10
```

NOTE: `run_mlperf_trial` automatically creates `$RESULT_DIR/_trial_start_final_profile` as a timestamp marker before launching training. `discover_trace()` uses this marker (not the raw log) for `-newer` comparison to avoid the timestamp race.

Restore profiling config:

```python
overrides["profile"] = False
overrides["use_pytorch_profiler"] = True
overrides["profile_step_start"] = 60
overrides["profile_step_end"] = 61
overrides["tensorboard_dir"] = "/workspace/code/tensorboard"
overrides["torch_profiler_use_gzip"] = False
with open(config_path, "w") as f:
    yaml.dump(config, f, default_flow_style=False)
```

Locate, copy, and filter the trace following the canonical procedure from
[`actions/profile.md`](profile.md) Step 4, with these substitutions for the final-profile run:

- `RAW_LOG="$RESULT_DIR/attempt_final_profile_raw.log"`
- Copy/decompress destination: `$RESULT_DIR/final_trace.json`
- Filter destination: `$RESULT_DIR/final_filtered_trace.json`
- Marker file: `${TRIAL_START_MARKER:-$RESULT_DIR/_trial_start_final_profile}`

Run the same detect → search (`$ACTUAL_TB_DIR`, then `$TB_TRACE_DIR`) → `discover_trace()`
fallback → `filter_trace` pipeline from profile.md with these variable values. Skip
gzip/validation steps that are specific to the baseline run; Step 5 below re-gzips and
validates the final trace on demand.

### Step 5: TraceLens CLI Comparative Analysis

If both baseline and final traces exist, run TraceLens CLI on the final trace and compare against the baseline analysis; otherwise skip and note unavailability in the report.

**Ensure TraceLens CLI is installed:**
```bash
TraceLens_generate_perf_report_pytorch --help >/dev/null 2>&1 || \
  (cp -r /hyperloom/TraceLens-internal /tmp/TraceLens-internal && pip install -e /tmp/TraceLens-internal)
```

**Run TraceLens CLI on final trace:**
```bash
gzip -kf "$RESULT_DIR/final_filtered_trace.json" 2>/dev/null || true

mkdir -p "$RESULT_DIR/tracelens_output/final/perf_report_csvs"
TraceLens_generate_perf_report_pytorch \
  --profile_json_path "$RESULT_DIR/final_filtered_trace.json.gz" \
  --output_csvs_dir "$RESULT_DIR/tracelens_output/final/perf_report_csvs" \
  --gpu_arch_json_path /hyperloom/TraceLens-internal/TraceLens/AgenticMode/Standalone/utils/arch/MI355X.json \
  --enable_pseudo_ops

PYTHONPATH="/hyperloom/TraceLens-internal:$PYTHONPATH" \
python3 /hyperloom/TraceLens-internal/TraceLens/AgenticMode/Standalone/orchestrator_prepare.py \
  --trace-path "$RESULT_DIR/final_filtered_trace.json.gz" \
  --platform MI355X \
  --output-dir "$RESULT_DIR/tracelens_output/final"
```

**Compare baseline vs final:**
```python
import json, csv, os

def load_gpu_timeline(tl_dir):
    path = os.path.join(tl_dir, "perf_report_csvs", "gpu_timeline.csv")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return {row["type"]: float(row["percent"]) for row in csv.DictReader(f)}

def load_category_summary(tl_dir):
    path = os.path.join(tl_dir, "perf_report_csvs", "ops_summary_by_category.csv")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return {row["op category"]: float(row["Percentage (%)"]) for row in csv.DictReader(f)}

baseline_dir = f"{RESULT_DIR}/tracelens_output/baseline"
final_dir = f"{RESULT_DIR}/tracelens_output/final"

baseline_timeline = load_gpu_timeline(baseline_dir)
final_timeline = load_gpu_timeline(final_dir)
baseline_cats = load_category_summary(baseline_dir)
final_cats = load_category_summary(final_dir)
```

### Step 6: Write optimization report

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

## TraceLens Analysis

### Baseline vs Optimized Kernel Breakdown

| Kernel | Baseline GPU% | Optimized GPU% | Delta | Bound Type |
|--------|--------------|----------------|-------|------------|
| (from TraceLens comparative analysis) |

### Communication-Compute Overlap

| Metric | Baseline | Optimized | Change |
|--------|----------|-----------|--------|
| Comm-compute overlap | X% | Y% | +Z% |
| MFMA utilization | X% | Y% | +Z% |
| Memory BW utilization | X% | Y% | +Z% |

### Kernel Changes
1. (kernel with largest time reduction)
2. ...

### Remaining Optimization Potential
- (areas TraceLens identifies as still suboptimal)

## Recommendations for Production
- ...

## Reproducibility

Baseline command pattern: see [`REFERENCE.md`](../REFERENCE.md) § MLPerf Run Commands →
"Local Run".

**Optimized command:** identical to the baseline command with the kept `export <VAR>=<value>`
overrides inserted between `source config_MI355X_1x8x1_fp8.sh` and the second
`source ... && bash run_and_time.sh`. List the actual applied env vars here (GBS, LR,
fusion flags, FP8 recipe, NCCL buffers, etc.) so the run can be replayed verbatim.
```

### Step 7: Ingest key findings into KB

```bash
python3 $SKILL_ROOT/kb/kb_ingest.py \
    --category "final_result" \
    --model "GPT-OSS-20B" \
    --action "Full Tier 4 convergence run with all optimizations" \
    --lesson "TTT=${TTT_SECONDS}s, status=${TTT_STATUS}, final_eval_loss=${FINAL_EVAL_LOSS}, ms_per_iter=${FINAL_MS}" \
    --tags "mlperf,training,MI355X,gpt-oss-20b,tier3,convergence" \
    --gain "$CUMULATIVE_GAIN_PCT" \
    --status "$TTT_STATUS"
```

## Outputs

- `$RESULT_DIR/optimization_report.md` (TTT and narrative); `attempt_final_raw.log` / `attempt_final.log` (Tier 4); `final_trace.json`; TraceLens CLI output in `tracelens_output/final/` when run; KB ingest entries.

## Heuristic Update

N/A — terminal action.

## Failure Handling

- **`run_stop` aborted:** report eval_loss vs 3.34, estimate iterations from loss slope, check Tier 2 trends for convergence regression; add “Why Target Was Not Reached” if needed.
- **Crash:** report best-so-far config and crash analysis.
- **Hang (>30 min no progress):** inspect GPU utilization; do not kill—wait for recovery or natural failure.
- **Profiling (Step 4) fails:** ship report without trace comparison; state unavailable.
- **TraceLens CLI not installed or fails:** omit TraceLens section; note unavailability in report.
