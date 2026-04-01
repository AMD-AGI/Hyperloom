# Action: Generate Optimization Report

## Overview

Generates the final optimization report and ingests key findings into the KB.

## Inputs
- `$RESULT_DIR` with `results.tsv`, traces, logs
- All state from the optimization run

## Procedure

### Step 1: Re-profile final optimized config

```bash
pkill -9 -f "primus/cli/main.py" 2>/dev/null; sleep 5
MASTER_PORT=$((MASTER_PORT + 1))

torchrun --nproc_per_node=$NUM_GPUS --master_port=$MASTER_PORT \
  -m primus.cli.main train pretrain \
  --config "$CONFIG_YAML" \
  $KEPT_OVERRIDES \
  profile=true use_pytorch_profiler=true \
  profile_step_start=6 profile_step_end=7 \
  2>&1 | tee "$RESULT_DIR/final_profile.log"
```

Copy the trace to `$RESULT_DIR/final_trace.json`.

### Step 2: Run TraceLens comparative analysis

```
Tool: run_comparative_analysis
Arguments:
  gpu1_kineto: $RESULT_DIR/baseline_trace.json
  gpu2_kineto: $RESULT_DIR/final_trace.json
  gpu1_name: "baseline"
  gpu2_name: "optimized"
  cleanup: false
```

### Step 3: Write optimization report

Write to `$RESULT_DIR/optimization_report.md`:

```markdown
# <Model> Training Optimization Report — <GPU> <N>-GPU

**Date:** YYYY-MM-DD
**Platform:** N× AMD <GPU>
**Container:** `<image>`
**Commit:** `<hash>`
**Model:** <description>

---

## Executive Summary

| Metric | Baseline | Optimized | Delta |
|---|---|---|---|
| **ms / iter** | X | Y | **−Z ms (−W%)** |
| **Samples/sec/GPU** | A | B | **+C%** |
| Global batch size | GBS | GBS | unchanged |
| Kept optimizations | — | M of N | |
| Discarded / crashed | — | K of N | |

**Final config overrides:**
```
override1=value1
override2=value2
```

**Code patches kept:** <description>

---

## All Attempts

| # | ms/iter | Δ vs baseline | Status | Description |
|---|---|---|---|---|
| 0 | X | — | baseline | ... |
| 1 | Y | −Z% | keep | ... |
| ... |

## What Worked
- ...

## What Didn't Work
- ...

## GEAK Kernel Optimization (if used)
| Kernel | GPU Time % | GEAK Task ID | Result |
|--------|-----------|--------------|--------|
| ... | ... | ... | keep/discard |

## Kernel Profile Comparison
(baseline vs final top-20 kernels)

## TraceLens Analysis
(GPU timeline breakdown, ops summary by category, key findings)

## Sweep Results (if run)
(MBS × precision Pareto table)

## Recommendations for Production
- ...

## Reproducibility
**Baseline command:**
```bash
torchrun --nproc_per_node=N --master_port=29500 \
  -m primus.cli.main train pretrain \
  --config <config> \
  profile=false use_pytorch_profiler=false
```

**Optimized command:**
```bash
torchrun --nproc_per_node=N --master_port=29500 \
  -m primus.cli.main train pretrain \
  --config <config> \
  <all kept overrides> \
  profile=false use_pytorch_profiler=false
```
```

### Step 4: Ingest key findings into KB

```bash
# For each kept optimization:
python3 $SKILL_ROOT/kb/kb_ingest.py \
    --category "fusion_flags" \
    --model "$MODEL_NAME" \
    --action "$ACTION_DESCRIPTION" \
    --lesson "$KEY_FINDING" \
    --tags "training,$MODEL_CLASS,$GPU_TYPE" \
    --gain "$GAIN_PCT" \
    --status "keep"

# For each failed optimization:
python3 $SKILL_ROOT/kb/kb_ingest.py \
    --category "pitfall" \
    --model "$MODEL_NAME" \
    --action "$ACTION_DESCRIPTION" \
    --lesson "$WHY_IT_FAILED" \
    --tags "training,$MODEL_CLASS,$GPU_TYPE" \
    --gain "0" \
    --status "discard"
```

## Outputs
- `$RESULT_DIR/optimization_report.md`: comprehensive report
- `$RESULT_DIR/final_trace.json`: final profiler trace
- `$RESULT_DIR/tracelens_output/`: TraceLens comparative analysis
- KB entries for all key findings

## Failure Handling
- If final profiling fails: write report without trace comparison
- If TraceLens unavailable: skip TraceLens section, note in report
