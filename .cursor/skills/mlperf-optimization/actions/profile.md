# Action: Profile and Diagnose Bottlenecks

## Inputs
- `$MLPERF_DIR`, `$CONFIG_SH`, `$EXP`
- `$RESULT_DIR` for storing traces

## Procedure

### Step 1: Modify config for profiling

Edit the YAML to enable profiling temporarily:

```python
import yaml

config_path = EXP
with open(config_path) as f:
    config = yaml.safe_load(f)

overrides = config["modules"]["pre_trainer"]["overrides"]
overrides["profile"] = True
overrides["use_pytorch_profiler"] = True
overrides["profile_step_start"] = 6
overrides["profile_step_end"] = 7

with open(config_path, "w") as f:
    yaml.dump(config, f, default_flow_style=False)
```

### Step 2: Run profiling training pass (Tier 1)

```bash
source "$SKILL_ROOT/scripts/common.sh"
run_mlperf_trial "profile" 1 10
```

The raw log is at `$RESULT_DIR/attempt_profile_raw.log`.

### Step 3: Restore config (disable profiling)

```python
overrides["profile"] = False
overrides["use_pytorch_profiler"] = True
overrides["profile_step_start"] = 60
overrides["profile_step_end"] = 61
with open(config_path, "w") as f:
    yaml.dump(config, f, default_flow_style=False)
```

### Step 4: Locate and copy trace files

```bash
TRACE_FILE=$(find /workspace /tmp /root -name "*.pt.trace.json" -newer "$RESULT_DIR/profile.log" 2>/dev/null | head -1)
if [ -n "$TRACE_FILE" ]; then
    cp "$TRACE_FILE" "$RESULT_DIR/baseline_trace.json"
fi
```

### Step 5: Kernel breakdown analysis

```python
import json
from collections import defaultdict

with open(f"{RESULT_DIR}/baseline_trace.json") as f:
    trace = json.load(f)

gpu_events = [e for e in trace["traceEvents"]
              if e.get("cat") == "kernel" and "dur" in e]

kernel_time = defaultdict(float)
kernel_count = defaultdict(int)
for e in gpu_events:
    kernel_time[e["name"]] += e["dur"]
    kernel_count[e["name"]] += 1

total = sum(kernel_time.values())
print(f"\nTop-20 GPU kernels ({total/1e6:.1f}s total):")
for name, t in sorted(kernel_time.items(), key=lambda x: -x[1])[:20]:
    print(f"  {name[:70]:70s}  {t/1000:>8.1f}ms  {t/total*100:>5.1f}%  {kernel_count[name]:>4d}x")
```

### Step 6: Categorize kernel breakdown

```python
categories = {
    "gemm": 0, "attention": 0, "moe_dispatch": 0,
    "communication": 0, "elementwise": 0, "fp8_ops": 0, "other": 0
}

for name, t in kernel_time.items():
    pct = t / total * 100
    if "Cijk_" in name or "hipblas" in name.lower():
        categories["gemm"] += pct
    elif "fmha" in name or "attn" in name.lower() or "flash" in name.lower():
        categories["attention"] += pct
    elif "permute" in name or "scatter" in name or "moe" in name.lower():
        categories["moe_dispatch"] += pct
    elif "nccl" in name.lower() or "allreduce" in name.lower() or "alltoall" in name.lower():
        categories["communication"] += pct
    elif "cast_transpose" in name or "fp8" in name.lower() or "amax" in name.lower():
        categories["fp8_ops"] += pct
    elif "elementwise" in name or "vectorized" in name:
        categories["elementwise"] += pct
    else:
        categories["other"] += pct
```

### Step 7: Identify GEAK candidates

```python
geak_candidates = []
for name, t in sorted(kernel_time.items(), key=lambda x: -x[1])[:20]:
    pct = t / total * 100
    if pct < 2.0:
        continue
    if "Cijk_" in name:
        continue  # vendor BLAS
    if "aiter::" in name:
        continue  # vendor attention
    if "nccl" in name.lower():
        continue  # communication
    geak_candidates.append({"name": name, "gpu_pct": pct, "count": kernel_count[name]})
```

### Step 8: Adjust heuristic based on profile

```python
if categories["gemm"] > 60:
    priors["fusion-flags"] *= 0.7
    priors["kernel-opt"] *= 0.5

if categories["moe_dispatch"] > 5:
    priors["fusion-flags"] *= 1.5

if categories["communication"] > 15:
    priors["parallelism"] *= 1.3

if categories["fp8_ops"] > 5:
    # FP8 cast/transpose overhead — check TE knobs
    priors["params"] *= 1.2

if categories["elementwise"] > 5:
    priors["kernel-opt"] *= 1.3
```

## Outputs
- `$RESULT_DIR/baseline_trace.json`: profiler trace
- Kernel breakdown by category
- `geak_candidates`: list of kernels eligible for GEAK optimization
- Updated heuristic priors based on profile

## Failure Handling
- If no trace produced: check profile=true was set in YAML, check output directory
- If profiling crashes: reduce profile_step_end, run fewer iters
- If trace too large: filter using scripts/common.sh:filter_trace()
