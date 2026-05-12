# Action: Optimization Report

Generate the final optimization report and contribute new knowledge to the KB.

## Inputs
- All results from previous actions
- `results.tsv` from sweep
- Per-kernel OOB results
- Baseline and optimized throughput numbers
- Target comparison data (if available)

## Procedure

### Write optimization report to `/workspace/hyperloom/optimization_report.md`

Remote-mode path ownership is strict:

- Full RayJob runtime artifacts may remain under `/wekafs/...` (`$RESULT_DIR`,
  `$TRACE_DIR`, OOB workspaces, sweep directories, TraceLens CSVs).
- The final user-facing report bundle MUST be written under
  `/workspace/hyperloom/`, because Claw persists/uploads this directory to S3 at
  session end.
- If report generation runs inside the RayJob, first write the detailed raw
  artifacts to `/wekafs/...`, then copy or summarize the final report,
  `ci_metrics.json`, result index, and manifest into `/workspace/hyperloom/`.
- If report generation runs in the sandbox, treat `/wekafs/...` as read-only
  input and write all generated files only under `/workspace/hyperloom/`.

```markdown
# Inference Optimization Report — {Model Name}

## Executive Summary
- **Model**: {model_id} ({param_count})
- **Hardware**: {gpu_count}x {gpu_type} ({num_nodes} nodes)
- **Mode**: Remote (SaFE RayJob, {num_nodes} nodes, TP={tp})
- **Framework**: {framework} v{version}
- **Optimization attempts**: {N} ({kept} kept, {discarded} discarded, {crashed} crashed)
- **Total improvement**: {total_pct}% (baseline → optimized)
- **Best output throughput**: {best_tput} tok/s at CONC={conc}
- **Best tok/s/GPU**: {best_tput_per_gpu} at CONC={conc}

## TraceLens Bottleneck Analysis (Baseline)
| Category | GPU Time (ms) | % | OOB Candidate? |
|----------|--------------|---|-----------------|

## Backend Exploration Results
| Backend/Scheduling Flag | Individual Gain | Combined Gain | Status |
|------------------------|----------------|---------------|--------|

## Server Parameter Tuning
| Parameter | Gain | Status |
|-----------|------|--------|

## OOB Kernel/Code Optimization
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
- `/workspace/hyperloom/optimization_report.md`
- `/workspace/hyperloom/ci_metrics.json` if CI metrics are requested
- `/workspace/hyperloom/MANIFEST.txt` or equivalent index pointing to raw
  `/wekafs/...` RayJob artifacts
- New KB entries for key findings
- Conflict resolutions (if any) logged to `kb/conflicts.jsonl`

## Heuristic Update
N/A — terminal action.

**Remote mode:** After the report is generated, stop the RayJob and clean up any parallel
sweep workloads. See [`../modes/REMOTE.md`](../modes/REMOTE.md) "Cleanup" section.

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
