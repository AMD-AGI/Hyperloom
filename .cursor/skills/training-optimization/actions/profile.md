# Action: Profile and Diagnose Bottlenecks

## Inputs
- `$CONFIG_YAML`, `$NUM_GPUS`, `$PRIMUS_ROOT`, `$MASTER_PORT`
- `$RESULT_DIR` for storing traces

## Procedure

### Step 1: Run profiling training pass

```bash
pkill -9 -f "primus/cli/main.py" 2>/dev/null || true
sleep 5

MASTER_PORT=$((MASTER_PORT + 1))

cd "$PRIMUS_ROOT"
torchrun --nproc_per_node=$NUM_GPUS --master_port=$MASTER_PORT \
  -m primus.cli.main train pretrain \
  --config "$CONFIG_YAML" \
  profile=true use_pytorch_profiler=true \
  profile_step_start=6 profile_step_end=7 \
  2>&1 | tee "$RESULT_DIR/profile.log"
```

This produces `.pt.trace.json` files in the training output directory.

### Step 2: Locate trace files

```bash
TRACE_DIR=$(find "$PRIMUS_ROOT" /tmp -name "*.pt.trace.json" -newer "$RESULT_DIR/profile.log" -printf "%h\n" 2>/dev/null | head -1)
# Or check the workspace path from config
TRACE_FILE=$(find "$TRACE_DIR" -name "*.pt.trace.json" | head -1)
cp "$TRACE_FILE" "$RESULT_DIR/baseline_trace.json" 2>/dev/null || true
```

### Step 3: Kernel breakdown analysis

```python
import json
from collections import defaultdict

with open(TRACE_FILE) as f:
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

### Step 4: Categorize kernel breakdown

```python
categories = {
    "gemm": 0, "attention": 0, "moe_dispatch": 0,
    "communication": 0, "elementwise": 0, "other": 0
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
    elif "elementwise" in name or "vectorized" in name:
        categories["elementwise"] += pct
    else:
        categories["other"] += pct
```

### Step 5: REQUIRED — TraceLens analysis (CLI)

TraceLens is **mandatory** for every profile. Use the TraceLens CLI tools directly.

**Ensure TraceLens CLI is installed:**
```bash
TraceLens_generate_perf_report_pytorch --help >/dev/null 2>&1 || \
  pip install /hyperloom/TraceLens-internal 2>/dev/null || \
  (cp -r /hyperloom/TraceLens-internal /tmp/TraceLens-internal && pip install -e /tmp/TraceLens-internal)
```

**Generate performance report:**
```bash
mkdir -p "$RESULT_DIR/tracelens_output/baseline"
TraceLens_generate_perf_report_pytorch \
  --profile_json_path "$TRACE_FILE" \
  --output_xlsx_path "$RESULT_DIR/tracelens_output/baseline/perf_report.xlsx" \
  --output_csvs_dir "$RESULT_DIR/tracelens_output/baseline/perf_report_csvs" \
  --gpu_arch_json_path /hyperloom/TraceLens-internal/TraceLens/AgenticMode/Standalone/utils/arch/MI355X.json \
  --enable_pseudo_ops \
  --group_by_num_kernels
```

**Prepare category data (GPU utilization, top ops, tree data, category filtering):**
```bash
python3 /hyperloom/TraceLens-internal/TraceLens/AgenticMode/Standalone/orchestrator_prepare.py \
  --trace-path "$TRACE_FILE" \
  --platform MI355X \
  --output-dir "$RESULT_DIR/tracelens_output/baseline"
```

**Run standalone analysis subagents:**

Read the skill file `/hyperloom/TraceLens-internal/TraceLens/AgenticMode/Standalone/.cursor/skills/standalone-analysis-orchestrator.md` and follow Steps 6–10 (system-level analysis, compute kernel analysis, validation, aggregation, and report generation) using:
- Output directory: `$RESULT_DIR/tracelens_output/baseline`
- Platform: `MI355X`
- Analysis mode: `default`

The final standalone analysis report will be at `$RESULT_DIR/tracelens_output/baseline/standalone_analysis.md`.

### Step 6: Identify GEAK candidates

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
    if "FmhaBwd" in name:
        continue  # CK library
    if "nccl" in name.lower():
        continue  # communication
    geak_candidates.append({"name": name, "gpu_pct": pct, "count": kernel_count[name]})
```

### Step 7: Build action stack from profile

```python
# Adjust heuristic based on profile
if categories["gemm"] > 60:
    # GEMM-dominated: limited room for config/code gains
    priors["fusion-flags"] *= 0.7
    priors["kernel-opt"] *= 0.5

if categories["moe_dispatch"] > 5:
    # Significant MoE dispatch overhead — fusion flags can help
    priors["fusion-flags"] *= 1.5

if categories["communication"] > 15:
    # Communication-heavy — parallelism or NCCL tuning
    priors["parallelism"] *= 1.3

if categories["elementwise"] > 5:
    # Unfused elementwise — torch.compile or GEAK
    priors["kernel-opt"] *= 1.3
```

## Outputs
- `$RESULT_DIR/baseline_trace.json`: profiler trace
- `$RESULT_DIR/tracelens_output/baseline/`: TraceLens analysis
- Kernel breakdown by category (gemm, attention, moe_dispatch, communication, elementwise)
- `geak_candidates`: list of kernels eligible for GEAK optimization
- Updated heuristic priors based on profile

## Failure Handling
- If no trace produced: check profile=true was set, check output directory
- If trace too large: filter using `scripts/common.sh:filter_trace()`
- TraceLens CLI not installed: run `pip install /hyperloom/TraceLens-internal` (NFS fallback: copy to `/tmp` first)
- TraceLens CLI fails: fall back to manual kernel analysis (Step 3)
