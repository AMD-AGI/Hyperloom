# Action: Re-Profile (Iterative Bottleneck Refresh)

## Overview

After major optimizations, GPU time shifts: former minor kernels can dominate and new GEAK candidates may appear. Re-profiling refreshes the bottleneck landscape so downstream DFS steps target the current profile.

## Inputs

`kept_*` optimizations, `kernel_candidates`, `RESULT_DIR/categories.json`, `state["tracelens_latest_output"]`, reprofile index `N`.

**When:** gain > 2%; all fusion flags done (#4); kept kernel-opt (#5). Stack: `score = max(remaining_scores) * 0.8` (not independently scored).

## Procedure

### Step 1: Run profiling trial with current optimizations

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
REPROFILE_N=$((state_reprofile_count + 1))
run_mlperf_trial "reprofile_${REPROFILE_N}" 1 10
```

NOTE: `run_mlperf_trial` automatically creates `$RESULT_DIR/_trial_start_reprofile_${REPROFILE_N}` as a timestamp marker before launching training. `discover_trace()` uses this marker (not the raw log) for `-newer` comparison to avoid the timestamp race.

Restore profiling off for normal runs:

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

### Step 2: Locate, filter, and validate trace

Detect the actual tensorboard path from the training log (the framework ignores the `tensorboard_dir` YAML setting and logs the real path at `args.tensorboard_dir is deprecated, the tensorboard path is: <path>`). Search the detected dir first, then our configured `tb_traces`, then fall back to `discover_trace()`:

```bash
source "$SKILL_ROOT/scripts/common.sh"
TB_TRACE_DIR="$RESULT_DIR/tb_traces"
TRACE_FOUND=""
RAW_LOG="$RESULT_DIR/attempt_reprofile_${REPROFILE_N}_raw.log"

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
        python3 -c "
import gzip, json
with gzip.open('$FOUND_GZ', 'rt') as f:
    trace = json.load(f)
with open('$RESULT_DIR/reprofile_${REPROFILE_N}_trace.json', 'w') as f:
    json.dump(trace, f)
"
        TRACE_FOUND="pytorch_json_gz"
        break
    elif [ -n "$FOUND_JSON" ]; then
        cp "$FOUND_JSON" "$RESULT_DIR/reprofile_${REPROFILE_N}_trace.json"
        TRACE_FOUND="pytorch_json"
        break
    fi
done

# Fallback: discover_trace() with marker file and hint dir
if [ -z "$TRACE_FOUND" ]; then
    MARKER="${TRIAL_START_MARKER:-$RAW_LOG}"
    HINT_DIR="${ACTUAL_TB_DIR:-$TB_TRACE_DIR}"
    TRACE_INFO=$(discover_trace "$MARKER" \
        "$RESULT_DIR/reprofile_${REPROFILE_N}_trace.json" "$HINT_DIR" 2>&1)
    TRACE_FOUND=$(echo "$TRACE_INFO" | grep "^TRACE_FORMAT=" | cut -d= -f2)
fi

if [ -n "$TRACE_FOUND" ]; then
    filter_trace "$RESULT_DIR/reprofile_${REPROFILE_N}_trace.json" \
                 "$RESULT_DIR/reprofile_${REPROFILE_N}_filtered.json"
    gzip -kf "$RESULT_DIR/reprofile_${REPROFILE_N}_filtered.json"
fi
```

### Step 3: TraceLens (optional)

`run_full_standalone_analysis` on `reprofile_${REPROFILE_N}_filtered.json.gz` → `tracelens_output/reprofile_${REPROFILE_N}` (`MI355X`, `pytorch` vs `rocprofv3` from `TRACE_FORMAT`); else skip.

### Step 4: Parse trace and compare categories

```bash
python3 "$SKILL_ROOT/scripts/parse_trace.py" \
    --trace-path "$RESULT_DIR/reprofile_${REPROFILE_N}_trace.json" \
    --result-dir "$RESULT_DIR/reprofile_${REPROFILE_N}"

# Compare against previous profile
python3 -c "
import json
new = json.load(open('$RESULT_DIR/reprofile_${REPROFILE_N}/categories.json'))
old = json.load(open('$RESULT_DIR/categories.json'))
for cat in new:
    delta = new[cat] - old.get(cat, 0)
    if abs(delta) > 1.0:
        print(f'{cat}: {old.get(cat,0):.1f}% -> {new[cat]:.1f}% (delta={delta:+.1f}%)')
"
```

If TraceLens ran, compare overlap / MFMA / bound mix vs prior.

### Step 5: Update action stack and kernel candidates

Merge `geak_candidates.json` from the new parse (and TraceLens state); boost `kernel-opt` when new candidates exist:

```python
import json

rd = f"{RESULT_DIR}/reprofile_{REPROFILE_N}"
new_cands = json.load(open(f"{rd}/geak_candidates.json"))
existing = {c["name"] for c in state["kernel_candidates"]}
for c in new_cands:
    if c["name"] not in existing:
        state["kernel_candidates"].append({**c, "source": "re-profile"})
state["tracelens_latest_output"] = f"{RESULT_DIR}/tracelens_output/reprofile_{REPROFILE_N}"
if new_cands:
    priors["kernel-opt"] *= 1.3
new_tl = parse_tracelens_output(state["tracelens_latest_output"])
if new_tl["comm_compute_overlap"] < state["comm_compute_overlap"]:
    priors["comm-overlap"] *= 1.2
state["comm_compute_overlap"] = new_tl["comm_compute_overlap"]
state["mfma_utilization"] = new_tl["mfma_utilization"]
```

Promote `reprofile_N/categories.json` to `RESULT_DIR/categories.json` as the new baseline when appropriate.

## Outputs

New trace + `reprofile_N/{profiler_summary,categories,geak_candidates,kernel_breakdown}.json`; optional `tracelens_output/reprofile_N/`; updated `kernel_candidates`, priors, TraceLens state; Step 4 category deltas.

## Heuristic Update

N/A — re-profile is triggered by score update rules, not scored independently.
New GEAK candidates boost kernel-opt by 1.3x.

## Failure Handling

Profiling failure → warn, skip re-profile, continue with prior candidates. No TraceLens → `parse_trace.py` only. All category deltas < 1% → log stable. Max 3 re-profiles per run.
