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

If the prompt specifies a `ci_metrics.json` output path, write it with a **flat**
top-level schema. The CI report parser (`report_generator.py`) requires these
**EXACT field names** — using different names (e.g. `baseline_output_tput_tok_s`)
will cause gain=N/A in the CI report.

```json
{
  "baseline_throughput": <total output tok/s across all GPUs>,
  "optimized_throughput": <total output tok/s across all GPUs>,
  "gain_pct": <float, 0.0 if no improvement>,
  "tok_per_gpu_baseline": <output tok/s per GPU>,
  "tok_per_gpu_optimized": <output tok/s per GPU>,
  "actions_taken": ["action1_description", "action2_description"]
}
```

**CRITICAL field name rules:**
- Use `baseline_throughput`, NOT `baseline_output_tput_tok_s`
- Use `tok_per_gpu_baseline`, NOT `baseline_output_tput_per_gpu`
- Use `optimized_throughput`, NOT `optimized_output_tput_tok_s`
- Use `tok_per_gpu_optimized`, NOT `optimized_output_tput_per_gpu`
- Use `gain_pct`, NOT `improvement_pct` or `total_improvement_pct`
- Use `actions_taken`, NOT `actions` or `optimization_actions`

Do NOT nest baseline/optimized into sub-objects. You may add extra metadata
fields (model, framework, gpu_type, etc.) but the six fields above MUST be
present with these exact names. If no optimization improved over baseline,
set `gain_pct` to `0.0` and optimized values equal to baseline.
