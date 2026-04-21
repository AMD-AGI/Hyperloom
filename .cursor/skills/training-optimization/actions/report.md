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

## Claw Mode: Session Outcome Self-Check (RCA)

Claw's v1 RCA pipeline consumes an optional outcome file written at end-of-turn.
When this skill is running inside the Claw sandbox, treat this as the **very last
step** — run it after the report (`$RESULT_DIR/optimization_report.md`) has been
finalized.

### Rules

- The skill is running inside the Claw sandbox iff `/workspace/.claw_message_id`
  exists. If it does not, exit this step silently (LOCAL / CI modes do not use RCA).
- Write an outcome file **only** when `$RESULT_DIR/optimization_report.md` is
  missing or clearly broken. On success, write **nothing** — backend treats
  absence-of-file as success.
- Do **not** emit a `task_outcome` value other than the three allowed ones.

### Allowed outcome values

| `task_outcome` | Meaning |
|----------------|---------|
| `completed_no_report` | `optimization_report.md` was not produced at all |
| `completed_broken_report` | Report file exists but is truncated / missing structure |
| `completed_invalid_report` | Report looks complete but key content is invalid |

### Script

Run this verbatim as the last step of `report`. It is a no-op on success and
on LOCAL / CI modes.

```bash
REPORT="${RESULT_DIR:?RESULT_DIR not set}/optimization_report.md"

MSG_ID_FILE="/workspace/.claw_message_id"
if [ -r "$MSG_ID_FILE" ]; then
  MID="$(tr -d '[:space:]' < "$MSG_ID_FILE")"
  if [ -n "$MID" ]; then
    OUT_DIR="/workspace/.claw_outcomes"
    mkdir -p "$OUT_DIR"

    outcome=""
    reason=""
    if [ ! -f "$REPORT" ]; then
      outcome="completed_no_report"
      reason="optimization_report.md was not produced"
    elif [ "$(wc -c < "$REPORT")" -lt 800 ]; then
      outcome="completed_broken_report"
      reason="report too small (<800 bytes)"
    elif [ "$(grep -c '^## ' "$REPORT")" -lt 2 ]; then
      outcome="completed_broken_report"
      reason="fewer than 2 H2 sections in report"
    elif ! grep -q '^| ' "$REPORT"; then
      outcome="completed_invalid_report"
      reason="no result tables in report"
    fi

    if [ -n "$outcome" ]; then
      python3 - "$OUT_DIR/$MID.json" "$outcome" "$reason" <<'PY'
import json, sys
path, outcome, reason = sys.argv[1], sys.argv[2], sys.argv[3]
with open(path, "w") as f:
    json.dump({"task_outcome": outcome, "reason": reason}, f)
PY
    fi
  fi
fi
```

Outcome file schema (forwarded by `executor-ts` to backend as `task_outcome` /
`outcome_reason` on the `executor-complete` callback):

```json
{"task_outcome": "completed_no_report",
 "reason": "optimization_report.md was not produced"}
```
