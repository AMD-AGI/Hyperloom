# Action: Profile and Diagnose Bottlenecks

## Overview

This action runs a short `torch.profiler` capture and TraceLens CLI analysis on Chrome JSON traces to locate bottlenecks (roofline, comm-compute overlap, stalls). Results feed **GEAK** candidate selection and **DFS** heuristic updates for subsequent trials.

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

### Step 6: TraceLens CLI — Install check and trace validation

**Ensure TraceLens CLI is installed:**
```bash
TraceLens_generate_perf_report_pytorch --help >/dev/null 2>&1 || \
  (cp -r /hyperloom/TraceLens-internal /tmp/TraceLens-internal && pip install -e /tmp/TraceLens-internal)
```

**Validate trace file:**
```bash
python3 -c "
import gzip, json, sys
path = '$RESULT_DIR/filtered_trace.json.gz'
try:
    with gzip.open(path) as f: data = json.load(f)
    print(f'Trace OK: {len(data.get(\"traceEvents\", []))} events')
except Exception as e:
    print(f'Trace validation failed: {e}', file=sys.stderr); sys.exit(1)
"
```

If validation fails, log a warning and continue with local-only analysis (Step 5). Do not block the pipeline.

### Step 7: TraceLens CLI — Generate performance report and category data

**Generate performance report:**
```bash
mkdir -p "$RESULT_DIR/tracelens_output/baseline/perf_report_csvs"
TraceLens_generate_perf_report_pytorch \
  --profile_json_path "$RESULT_DIR/filtered_trace.json.gz" \
  --output_csvs_dir "$RESULT_DIR/tracelens_output/baseline/perf_report_csvs" \
  --gpu_arch_json_path /hyperloom/TraceLens-internal/TraceLens/AgenticMode/Standalone/utils/arch/MI355X.json \
  --enable_pseudo_ops
```

**Prepare category data (GPU utilization, top ops, tree data, category filtering):**
```bash
PYTHONPATH="/hyperloom/TraceLens-internal:$PYTHONPATH" \
python3 /hyperloom/TraceLens-internal/TraceLens/AgenticMode/Standalone/orchestrator_prepare.py \
  --trace-path "$RESULT_DIR/filtered_trace.json.gz" \
  --platform MI355X \
  --output-dir "$RESULT_DIR/tracelens_output/baseline"
```

**Run standalone analysis subagents:**

Read the skill file `/hyperloom/TraceLens-internal/TraceLens/AgenticMode/Standalone/.cursor/skills/standalone-analysis-orchestrator.md` and follow Steps 6-10 (system-level analysis, compute kernel analysis, validation, aggregation, and report generation) using:
- Output directory: `$RESULT_DIR/tracelens_output/baseline`
- Platform: `MI355X`
- Analysis mode: `default`

The final standalone analysis report will be at `$RESULT_DIR/tracelens_output/baseline/standalone_analysis.md`.

### Step 8: Parse TraceLens CLI Results

Extract metrics from the CLI output files into a structured dict; save as `tracelens_metrics.json` for Step 10.

```python
import json, os

tl_dir = f"{RESULT_DIR}/tracelens_output/baseline"
manifest_path = os.path.join(tl_dir, "category_data", "category_manifest.json")

tracelens = {
    "roofline": {},             # per-kernel: "compute_bound" or "memory_bound"
    "comm_compute_overlap": 0,  # 0.0-1.0: fraction of comm hidden by compute
    "mfma_utilization": 0,      # 0.0-1.0: MFMA unit utilization
    "mem_bw_utilization": 0,    # 0.0-1.0: memory bandwidth utilization
    "operator_breakdown": {},   # operator_name -> {gpu_pct, category, bound_type}
    "eval_overhead_pct": 0,     # fraction of wall time spent in eval phases
    "stall_pct": 0,             # fraction of GPU time in stalls/sync
}

if os.path.exists(manifest_path):
    manifest = json.load(open(manifest_path))
    gpu_util = manifest.get("gpu_utilization", {})
    tracelens["mfma_utilization"] = gpu_util.get("computation_time_percent", 0) / 100
    tracelens["mem_bw_utilization"] = gpu_util.get("exposed_memcpy_time_percent", 0) / 100
    tracelens["comm_compute_overlap"] = 1.0 - (gpu_util.get("exposed_comm_time_percent", 0) / 100)
    tracelens["stall_pct"] = gpu_util.get("idle_time_percent", 0)
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
- TraceLens CLI not installed: `cp -r /hyperloom/TraceLens-internal /tmp/TraceLens-internal && pip install -e /tmp/TraceLens-internal`
- TraceLens CLI fails: fall back to Step 5 (`parse_trace.py`) only; `state["tracelens_baseline_output"] = None`; run `parse_trace.py --compute-heuristics` without `--tracelens` for category-only rules.
- Re-profile failure → warn, skip re-profile, continue with prior kernel candidates.
- Re-profile category deltas all < 1% → log `stable`, no prior adjustments.

## Re-Profile Trigger {#re-profile-trigger}

After major optimizations GPU time shifts — formerly minor kernels can dominate and
new GEAK candidates may appear. Re-profiling reruns the pipeline above against the
current optimized configuration so downstream DFS steps target the refreshed
bottleneck mix.

Re-profile is not scored as an independent DFS action; the orchestrator pushes it
onto the stack when Score Update Rules fire.

**When triggered (Score Update Rules):**
- Any kept action with gain > `REPROFILE_TRIGGER_PCT` (2.0%) (Rule #7).
- All fusion flags tested (Rule #4).
- A kernel-opt candidate was kept (Rule #5).

**Score:** `max(remaining_scores) × 0.8` (derived, never scored on its own).
**Budget:** max 3 re-profiles per run.

### Procedure

Re-run Steps 1 through 10 above, substituting every `"profile"` / `"baseline"`
label for `"reprofile_${N}"`, where `N = state["reprofile_count"] + 1`. All file
paths follow the same substitution, so the pipeline becomes:

- Raw log: `$RESULT_DIR/attempt_reprofile_${N}_raw.log`
- Trace copy/decompress destination: `$RESULT_DIR/reprofile_${N}_trace.json`
- Filtered trace: `$RESULT_DIR/reprofile_${N}_filtered.json` (+ `gzip -kf` for
  TraceLens consumption)
- `discover_trace` marker: `${TRIAL_START_MARKER:-$RESULT_DIR/_trial_start_reprofile_${N}}`
- TraceLens output directory: `$RESULT_DIR/tracelens_output/reprofile_${N}`
- `parse_trace.py --result-dir`: `$RESULT_DIR/reprofile_${N}`

`run_mlperf_trial "reprofile_${N}" 1 10` creates the marker automatically, so
`discover_trace()` uses the correct `-newer` reference even if the raw log
finishes writing after the trace file. Enable profiling in the YAML exactly as in
Step 1, run the trial, then restore profiling off for normal runs.

### Compare categories vs previous profile

```python
import json
new = json.load(open(f"$RESULT_DIR/reprofile_${N}/categories.json"))
old = json.load(open("$RESULT_DIR/categories.json"))
for cat in new:
    delta = new[cat] - old.get(cat, 0)
    if abs(delta) > 1.0:
        print(f"{cat}: {old.get(cat,0):.1f}% -> {new[cat]:.1f}% (delta={delta:+.1f}%)")
```

If TraceLens ran, compare overlap / MFMA / bound mix against the previous run in
`state["tracelens_latest_output"]`.

### Update action stack and kernel candidates

```python
import json, os

rd = f"{RESULT_DIR}/reprofile_{N}"
new_cands = json.load(open(f"{rd}/geak_candidates.json"))
existing = {c["name"] for c in state["kernel_candidates"]}
for c in new_cands:
    if c["name"] not in existing:
        state["kernel_candidates"].append({**c, "source": "re-profile"})
state["tracelens_latest_output"] = f"{RESULT_DIR}/tracelens_output/reprofile_{N}"
if new_cands:
    priors["kernel-opt"] *= 1.3

manifest_path = os.path.join(state["tracelens_latest_output"],
                             "category_data", "category_manifest.json")
if os.path.exists(manifest_path):
    manifest = json.load(open(manifest_path))
    gpu_util = manifest.get("gpu_utilization", {})
    new_overlap = 1.0 - (gpu_util.get("exposed_comm_time_percent", 0) / 100)
    new_mfma = gpu_util.get("computation_time_percent", 0) / 100
    if new_overlap < state["comm_compute_overlap"]:
        priors["comm-overlap"] *= 1.2
    state["comm_compute_overlap"] = new_overlap
    state["mfma_utilization"] = new_mfma
```

Promote `reprofile_${N}/categories.json` to `$RESULT_DIR/categories.json` as the
new baseline when appropriate (e.g., the orchestrator has committed the kept
optimization set).

### Re-Profile outputs

New trace + `reprofile_${N}/{profiler_summary,categories,geak_candidates,kernel_breakdown}.json`;
optional `tracelens_output/reprofile_${N}/`; updated `kernel_candidates`, priors,
and TraceLens state; category deltas from the compare step.
