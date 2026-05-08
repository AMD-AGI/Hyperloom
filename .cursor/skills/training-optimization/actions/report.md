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

### Step 2: Run TraceLens analysis on the optimized trace

Run the same TraceLens analysis as `actions/profile.md` Step 5, but pointing at the final/optimized trace and a separate output directory. The baseline run's `tracelens_output/baseline/` is reused for before/after comparison.

```bash
TL_DIR="/hyperloom/TraceLens-internal"
[ -d "/opt/TraceLens" ] && TL_DIR="/opt/TraceLens"
if ! command -v TraceLens_generate_perf_report_pytorch >/dev/null 2>&1; then
  if [ -d "$TL_DIR" ]; then
    cp -r "$TL_DIR" /tmp/TraceLens-internal && TL_DIR=/tmp/TraceLens-internal
  else
    git clone "${TRACELENS_GIT_URL:-https://github.com/AMD-AIG-AIMA/TraceLens-internal.git}" /tmp/TraceLens-internal && TL_DIR=/tmp/TraceLens-internal
  fi
  pip install -e "$TL_DIR"
fi
```

Read the Analysis Orchestrator skill file `$TL_DIR/TraceLens/Agent/Analysis/.cursor/skills/analysis-orchestrator.md` installed with TraceLens and strictly follow the instructions inside to perform the full agentic analysis workflow.

Inputs for this run:

| Orchestrator input | Value |
|---|---|
| `<trace_path>` | `$RESULT_DIR/final_trace.json` |
| `<platform>` | `$PLATFORM` (e.g. `MI355X`) |
| `<analysis_mode>` | `default` |
| `<output_dir>` | `$RESULT_DIR/tracelens_output/optimized` |

Requirements:
- Strictly follow the step order in the skill file — do not skip any steps.
- In Step 6 and Step 7, each category must be executed by an independent Task subagent to ensure context isolation.
- Each subagent must write out the corresponding findings file (`system_findings/` or `category_findings/`).
- Do not fabricate analysis results — all data must come from agent analysis output.
- Write the final report to `$RESULT_DIR/tracelens_output/optimized/analysis.md`.

Comparison: load `category_data/<cat>_findings.json` and `priority_data.json` from `tracelens_output/baseline/` and `tracelens_output/optimized/` and diff per-category GPU-time / efficiency to fill the "TraceLens Analysis" section of the report.

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
