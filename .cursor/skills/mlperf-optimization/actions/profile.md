# Action: Profile and Diagnose Bottlenecks

## Overview

This action runs a short `torch.profiler` capture and TraceLens standalone analysis on Chrome JSON traces to locate bottlenecks (roofline, comm–compute overlap, stalls). Results feed **GEAK** candidate selection and **DFS** heuristic updates for subsequent trials.

## Inputs

- `$MLPERF_DIR`, `$CONFIG_SH`, `$EXP`
- `$RESULT_DIR` for traces and analysis artifacts
- `$SKILL_ROOT` (`scripts/parse_trace.py`, `kb/kb_query.py`)

## KB Query

```bash
python3 $SKILL_ROOT/kb/kb_query.py "GPT-OSS-20B profiling TraceLens" --top-k 5 --compact
```

## Procedure

### Step 1: Modify config for profiling

Edit the YAML to enable profiling temporarily:

```python
import yaml, os

config_path = EXP
with open(config_path) as f:
    config = yaml.safe_load(f)

TB_TRACE_DIR = os.path.join(RESULT_DIR, "tb_traces")
os.makedirs(TB_TRACE_DIR, exist_ok=True)

overrides = config["modules"]["pre_trainer"]["overrides"]
overrides["profile"] = True
overrides["use_pytorch_profiler"] = True
overrides["profile_step_start"] = 5
overrides["profile_step_end"] = 8           # 3 active steps for statistical stability
overrides["tensorboard_dir"] = TB_TRACE_DIR  # deterministic output path
overrides["torch_profiler_use_gzip"] = True  # compress trace at source

with open(config_path, "w") as f:
    yaml.dump(config, f, default_flow_style=False)
```

### Step 2: Run profiling training pass (Tier 1)

```bash
source "$SKILL_ROOT/scripts/common.sh"
run_mlperf_trial "profile" 1 10
```

Raw log: `$RESULT_DIR/attempt_profile_raw.log`.

NOTE: `run_mlperf_trial` automatically creates `$RESULT_DIR/_trial_start_profile` as a timestamp marker before launching training. `discover_trace()` uses this marker (not the raw log) for `-newer` comparison to avoid the timestamp race where the trace file is written mid-run but the raw log finishes writing later.

### Step 3: Restore config (disable profiling)

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

### Step 4: Locate, copy, and filter trace files

Detect the actual tensorboard path from the training log (the framework ignores the `tensorboard_dir` YAML setting and logs the real path at `args.tensorboard_dir is deprecated, the tensorboard path is: <path>`). Search the detected dir first, then our configured `tb_traces`, then fall back to `discover_trace()`:

```bash
source "$SKILL_ROOT/scripts/common.sh"
TB_TRACE_DIR="$RESULT_DIR/tb_traces"
TRACE_FOUND=""
RAW_LOG="$RESULT_DIR/attempt_profile_raw.log"

# Detect actual tensorboard path from training log
ACTUAL_TB_DIR=""
if [ -f "$RAW_LOG" ]; then
    ACTUAL_TB_DIR=$(grep -oP 'the tensorboard path is: \K\S+' "$RAW_LOG" | tail -1)
fi

# Search: actual framework dir → configured TB_TRACE_DIR
for SEARCH_DIR in "$ACTUAL_TB_DIR" "$TB_TRACE_DIR"; do
    [ -z "$SEARCH_DIR" ] && continue
    [ -d "$SEARCH_DIR" ] || continue
    FOUND_GZ=$(find "$SEARCH_DIR" -name "*.pt.trace.json.gz" -type f 2>/dev/null \
        | xargs ls -t 2>/dev/null | head -1)
    FOUND_JSON=$(find "$SEARCH_DIR" -name "*.pt.trace.json" -type f 2>/dev/null \
        | xargs ls -t 2>/dev/null | head -1)
    if [ -n "$FOUND_GZ" ]; then
        echo "Torch profiler trace (gzip): $FOUND_GZ"
        python3 -c "
import gzip, json
with gzip.open('$FOUND_GZ', 'rt') as f:
    trace = json.load(f)
with open('$RESULT_DIR/baseline_trace.json', 'w') as f:
    json.dump(trace, f)
"
        TRACE_FOUND="pytorch_json_gz"
        break
    elif [ -n "$FOUND_JSON" ]; then
        echo "Torch profiler trace (json): $FOUND_JSON"
        cp "$FOUND_JSON" "$RESULT_DIR/baseline_trace.json"
        TRACE_FOUND="pytorch_json"
        break
    fi
done

# Fallback: discover_trace() with marker file and hint dir
if [ -z "$TRACE_FOUND" ]; then
    echo "No trace in known dirs, falling back to discover_trace()..."
    MARKER="${TRIAL_START_MARKER:-$RESULT_DIR/_trial_start_profile}"
    HINT_DIR="${ACTUAL_TB_DIR:-$TB_TRACE_DIR}"
    TRACE_INFO=$(discover_trace "$MARKER" \
        "$RESULT_DIR/baseline_trace.json" "$HINT_DIR" 2>&1)
    TRACE_FOUND=$(echo "$TRACE_INFO" | grep "^TRACE_FORMAT=" | cut -d= -f2)
fi

if [ -n "$TRACE_FOUND" ]; then
    echo "Trace found (format: $TRACE_FOUND)"
    filter_trace "$RESULT_DIR/baseline_trace.json" "$RESULT_DIR/filtered_trace.json"
else
    echo "WARNING: No trace found. Check profile=true in YAML and logs for rpdTracerControl."
fi
```

If `filtered_trace.json` exceeds 50MB, re-filter to `kernel` / `gpu_memcpy` / `user_annotation` only (`filtered_trace_compact.json`). Then:

```bash
gzip -kf "$RESULT_DIR/filtered_trace.json"
```

TraceLens input: `$RESULT_DIR/filtered_trace.json.gz`.

### Step 5: Parse trace (operator summary, categories, kernels, GEAK draft)

```bash
python3 "$SKILL_ROOT/scripts/parse_trace.py" \
    --trace-path "$RESULT_DIR/baseline_trace.json" \
    --result-dir "$RESULT_DIR"
```

Outputs: `profiler_summary.json`, `categories.json`, `kernel_breakdown.json`, `geak_candidates.json`.

### Step 6: TraceLens Trace Validation

```python
CallMcpTool(
    server="oci-traceLens-agent",
    toolName="check_trace_file",
    arguments={"trace_path": f"{RESULT_DIR}/filtered_trace.json.gz"}
)
```

If `check_trace_file` fails, log a warning and continue with local-only analysis (Step 5). Do not block the pipeline.

### Step 7: TraceLens Standalone Analysis

```python
CallMcpTool(
    server="oci-traceLens-agent",
    toolName="run_full_standalone_analysis",
    arguments={
        "trace_path": f"{RESULT_DIR}/filtered_trace.json.gz",
        "platform": "MI355X",
        "trace_type": "pytorch",
        "output_dir": f"{RESULT_DIR}/tracelens_output/baseline",
        "cleanup": False,
    }
)
```

### Step 8: Parse TraceLens Results

Extract metrics into a structured dict; save as `tracelens_metrics.json` for Step 10.

```python
tracelens = {
    "roofline": {},             # per-kernel: "compute_bound" or "memory_bound"
    "comm_compute_overlap": 0,  # 0.0-1.0: fraction of comm hidden by compute
    "mfma_utilization": 0,      # 0.0-1.0: MFMA unit utilization
    "mem_bw_utilization": 0,    # 0.0-1.0: memory bandwidth utilization
    "operator_breakdown": {},   # operator_name -> {gpu_pct, category, bound_type}
    "eval_overhead_pct": 0,     # fraction of wall time spent in eval phases
    "stall_pct": 0,             # fraction of GPU time in stalls/sync
}
```

### Step 9: Identify GEAK candidates

```python
MIN_CANDIDATES = 10
geak_candidates = []
for name, t in sorted(kernel_time.items(), key=lambda x: -x[1])[:30]:
    pct = t / total * 100
    if "Cijk_" in name:
        continue  # vendor BLAS
    if "aiter::" in name:
        continue  # vendor attention
    if "nccl" in name.lower():
        continue  # communication
    bound_type = tracelens["roofline"].get(name, "unknown")
    geak_candidates.append({
        "name": name, "gpu_pct": pct,
        "count": kernel_count[name], "bound_type": bound_type,
        "low_gpu_pct": pct < 2.0,  # advisory flag, not a filter
    })
# If fewer than MIN_CANDIDATES found in top-30, scan all remaining non-vendor kernels
if len(geak_candidates) < MIN_CANDIDATES:
    seen = {c["name"] for c in geak_candidates}
    for name, t in sorted(kernel_time.items(), key=lambda x: -x[1]):
        if name in seen:
            continue
        pct = t / total * 100
        if "Cijk_" in name or "aiter::" in name or "nccl" in name.lower():
            continue
        bound_type = tracelens["roofline"].get(name, "unknown")
        geak_candidates.append({
            "name": name, "gpu_pct": pct,
            "count": kernel_count[name], "bound_type": bound_type,
            "low_gpu_pct": pct < 2.0,
        })
        if len(geak_candidates) >= MIN_CANDIDATES:
            break
```

Rebuild `kernel_time` / `kernel_count` / `total` from `baseline_trace.json` if needed.

### Step 10: Compute heuristic adjustments

```bash
python3 "$SKILL_ROOT/scripts/parse_trace.py" \
    --compute-heuristics \
    --categories "$RESULT_DIR/categories.json" \
    --tracelens "$RESULT_DIR/tracelens_metrics.json"
```

Apply printed multipliers to DFS `priors` (optionally `> "$RESULT_DIR/heuristic_adjustments.json"`).

```python
memory_bound_pct = sum(
    v["gpu_pct"] for v in tracelens["operator_breakdown"].values()
    if v.get("bound_type") == "memory_bound"
)
state["comm_compute_overlap"] = tracelens["comm_compute_overlap"]
state["mfma_utilization"] = tracelens["mfma_utilization"]
state["memory_bound_kernel_pct"] = memory_bound_pct
state["tracelens_baseline_output"] = "$RESULT_DIR/tracelens_output/baseline"
state["tracelens_latest_output"] = state["tracelens_baseline_output"]
```

## Outputs

- Raw: `$RESULT_DIR/tb_traces/`; `baseline_trace.json`; `filtered_trace.json.gz`; `tracelens_output/baseline/`
- Parsed: `profiler_summary.json`, `categories.json`, `kernel_breakdown.json`, `geak_candidates.json`, `tracelens_metrics.json`
- Refined GEAK list with roofline `bound_type`; heuristic multipliers; orchestrator `state`

## Heuristic Update

- gemm > 60%: reduce fusion-flags (0.7x), kernel-opt (0.7x)
- moe_dispatch > 5%: boost fusion-flags (1.5x)
- communication > 15%: boost comm-tuning (1.3x)
- TraceLens overlap < 0.5: boost comm-tuning (1.5x)
- TraceLens overlap > 0.7: reduce comm-tuning priority (0.8x) — still runs, overlap is advisory
- memory_bound > 20%: boost kernel-opt (1.5x)
- fp8_ops > 3%: boost fp8-recipe-tuning (1.3x)
- kernel-opt score floor: 0.5x minimum (never suppressed below half)
- comm-tuning score floor: 0.5x minimum (never suppressed below half)
- Full rules in `scripts/parse_trace.py` `compute_heuristic_adjustments()`

## Failure Handling

- No trace: check `rpdTracerControl` in logs; `discover_trace()` for `.rpd` / `.pt.trace.json`; confirm `profile: true` and step range overlaps training.
- Profiling crashes: lower `profile_step_end`, fewer iters.
- Trace too large: `scripts/run_profile.sh` size-check / re-filter.
- RPD conversion fails: `sqlite3 <file>.rpd ".tables"`; install `rocmProfileData` for `rpd2tracing`.
- TraceLens MCP down: warn; continue with Step 5 only; `state["tracelens_baseline_output"] = None`; run `parse_trace.py --compute-heuristics` without `--tracelens` for category-only rules.
