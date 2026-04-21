# Action: Optimization Report

Generate the final optimization report and contribute new knowledge to the KB.

## Inputs
- All results from previous actions
- `results.tsv` from sweep
- Per-kernel GEAK results
- Baseline and optimized throughput numbers
- Target comparison data (if available)

## Procedure

### Write optimization report to `$WORK_DIR/optimization_report.md`

```markdown
# Inference Optimization Report — {Model Name}

## Executive Summary
- **Model**: {model_id} ({param_count})
- **Hardware**: {gpu_count}x {gpu_type} ({num_nodes} nodes)
- **Mode**: {Local | Claw (SaFE RayJob, {num_nodes} nodes, TP={tp})}
- **Framework**: {framework} v{version}
- **Optimization attempts**: {N} ({kept} kept, {discarded} discarded, {crashed} crashed)
- **Total improvement**: {total_pct}% (baseline → optimized)
- **Best output throughput**: {best_tput} tok/s at CONC={conc}
- **Best tok/s/GPU**: {best_tput_per_gpu} at CONC={conc}

## TraceLens Bottleneck Analysis (Baseline)
| Category | GPU Time (ms) | % | GEAK Candidate? |
|----------|--------------|---|-----------------|

## Backend Exploration Results
| Backend/Scheduling Flag | Individual Gain | Combined Gain | Status |
|------------------------|----------------|---------------|--------|

## Server Parameter Tuning
| Parameter | Gain | Status |
|-----------|------|--------|

## GEAK Kernel Optimization
| # | Kernel | GPU % | Kernel Speedup | Actual E2E | Status |
|---|--------|-------|---------------|-----------|--------|

## Target Comparison (if applicable)
| Metric | Target (e.g. NVIDIA B200) | Achieved (MI355X) | Gap |
|--------|--------------------------|-------------------|-----|

## Parameter Sweep (Optimized Version)
### ISL=1024 / OSL=1024
| CONC | Output tok/s | tok/s/GPU | TPOT (ms) | TTFT (ms) |
|------|-------------|-----------|-----------|-----------|

## Recommendations
1. ...
```

### Contribute knowledge to KB

After writing the report, ingest key findings into the KB. This is the **pre-hook** contribution (the post-optimization hook `knowledge-sink.py` handles automatic contribution).

```bash
python3 $SKILL_ROOT/kb/kb_ingest.py \
    --category backend_exploration \
    --model "$MODEL_NAME" \
    --framework "$FRAMEWORK" \
    --action "Summary: $BEST_OPTIMIZATION_DESCRIPTION" \
    --lesson "$KEY_LESSON" \
    --tags "$RELEVANT_TAGS" \
    --gain $TOTAL_GAIN_PCT --status KEEP \
    --context "Controlled A/B test, $(date +%Y-%m-%d)"
```

## Outputs
- `$WORK_DIR/optimization_report.md`
- New KB entries for key findings
- Conflict resolutions (if any) logged to `kb/conflicts.jsonl`

## Heuristic Update
N/A — terminal action.

**Claw mode:** After the report is generated, stop the RayJob and clean up any parallel
sweep workloads. See [`../modes/CLAW.md`](../modes/CLAW.md) "Cleanup" section.

### CI Mode: Write ci_metrics.json

If the prompt specifies a `ci_metrics.json` output path, copy this example
and change only the numbers:

```json
{"baseline_throughput": 8053.90, "optimized_throughput": 8850.12, "gain_pct": 9.88, "tok_per_gpu_baseline": 4026.95, "tok_per_gpu_optimized": 4425.06, "actions_taken": ["params_max_num_seqs_512", "kernel_fused_moe_kept"]}
```

How to compute:
- `baseline_throughput` = `output_throughput` from baseline benchmark JSON (total, all GPUs)
- `optimized_throughput` = `output_throughput` from final optimized benchmark JSON
- `tok_per_gpu_baseline` = `baseline_throughput / TP` (divide by TP, NOT total)
- `tok_per_gpu_optimized` = `optimized_throughput / TP`
- `gain_pct` = `(optimized - baseline) / baseline * 100`

The CI parser requires these exact six field names. Missing or renamed fields → N/A.

## Claw Mode: Session Outcome Self-Check (RCA)

Claw's v1 RCA pipeline consumes an optional outcome file written at end-of-turn.
When this skill is running inside the Claw sandbox, treat this as the **very last
step** — run it after the report is finalized **and** after RayJob cleanup
(`workload_stop`) is done.

### Rules

- The skill is running inside the Claw sandbox iff `/workspace/.claw_message_id`
  exists. If it does not, exit this step silently (LOCAL / CI modes do not use RCA).
- Write an outcome file **only** when the primary deliverable
  (`$WORK_DIR/optimization_report.md`) is missing or clearly broken.
- On success, write **nothing** — backend treats absence-of-file as success.
- Do **not** emit a `task_outcome` value other than the three allowed ones;
  anything else is ignored by the backend.

### Allowed outcome values

| `task_outcome` | Meaning |
|----------------|---------|
| `completed_no_report` | `optimization_report.md` was not produced at all |
| `completed_broken_report` | Report file exists but is truncated / missing structure |
| `completed_invalid_report` | Report looks complete but key content is invalid |

### Script

Run this verbatim as the final step of `report`. It is a no-op on success and
on LOCAL / CI modes.

```bash
REPORT="${WORK_DIR:?WORK_DIR not set}/optimization_report.md"

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

Outcome file schema (consumed by `executor-ts` and forwarded to backend as
`task_outcome` / `outcome_reason` on the `executor-complete` callback):

```json
{"task_outcome": "completed_no_report",
 "reason": "optimization_report.md was not produced"}
```
