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

### Auto-publish normalized results

After `optimization_report.md` and any available `ci_metrics.json` are written,
always attempt to normalize and publish the result directory. This is non-fatal:
if the results service is not reachable, keep the normalized files and continue
the final report normally.

```bash
HYPERLOOM_REPO="${HYPERLOOM_REPO:-}"
if [ -z "$HYPERLOOM_REPO" ]; then
  for candidate in \
    /workspace/Hyperloom \
    /wekafs/hyperloom/Hyperloom \
    /wekafs/zgong/Hyperloom \
    /opt/hyperloom; do
    if [ -f "$candidate/ci/publish_artifacts.py" ]; then
      HYPERLOOM_REPO="$candidate"
      break
    fi
  done
fi

if [ -n "$HYPERLOOM_REPO" ]; then
  python3 "$HYPERLOOM_REPO/ci/publish_artifacts.py" \
    --task-dir "${WORK_DIR:-/workspace/hyperloom}" \
    --out-dir "${WORK_DIR:-/workspace/hyperloom}/normalized" \
    --model "${MODEL_NAME:-${MODEL:-unknown}}" \
    --display-name "${DISPLAY_NAME:-hyperloom-auto-publish}" || true
else
  echo "Hyperloom publish_artifacts.py not found; skipping result publish"
fi
```

Default publish target:

```text
http://hyperloom-results-service.primus-claw-dev.svc.cluster.local
```

Override with `HYPERLOOM_RESULTS_SERVICE_URL` if needed.
