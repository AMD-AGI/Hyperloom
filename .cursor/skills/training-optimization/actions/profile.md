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

### Step 5: REQUIRED — TraceLens analysis

TraceLens is **mandatory** for every profile. This step runs the full agentic analysis workflow through TraceLens's own `analysis-orchestrator` skill. Do not call `TraceLens_generate_perf_report_pytorch*` or `orchestrator_prepare.py` directly — the orchestrator runs them internally.

**Resolve `analysis_mode`.** Training-optimization runs Megatron/Primus pretraining, so:

```bash
ANALYSIS_MODE=default   # uses TraceLens_generate_perf_report_pytorch internally
PLATFORM=MI355X         # adjust to actual GPU
```

**Ensure TraceLens is installed (with git-clone fallback):**

```bash
TL_DIR="/hyperloom/TraceLens-internal"
[ -d "/opt/TraceLens" ] && TL_DIR="/opt/TraceLens"
if ! command -v TraceLens_generate_perf_report_pytorch >/dev/null 2>&1; then
  if [ -d "$TL_DIR" ]; then
    cp -r "$TL_DIR" /tmp/TraceLens-internal && TL_DIR=/tmp/TraceLens-internal
  else
    git clone "${TRACELENS_GIT_URL:-https://github.com/AMD-AIG-AIMA/TraceLens-internal.git}" /tmp/TraceLens-internal && TL_DIR=/tmp/TraceLens-internal
  fi
  pip install -e "$TL_DIR"
fi
```

**Run the analysis orchestrator:**

Read the Analysis Orchestrator skill file `$TL_DIR/TraceLens/Agent/Analysis/.cursor/skills/analysis-orchestrator.md` installed with TraceLens and strictly follow the instructions inside to perform the full agentic analysis workflow.

Inputs for this run:

| Orchestrator input | Value |
|---|---|
| `<trace_path>` | `$TRACE_FILE` |
| `<platform>` | `$PLATFORM` (e.g. `MI355X`) |
| `<analysis_mode>` | `default` |
| `<output_dir>` | `$RESULT_DIR/tracelens_output/baseline` |

Requirements:
- Strictly follow the step order in the skill file — do not skip any steps.
- In Step 6 and Step 7, each category must be executed by an independent Task subagent to ensure context isolation.
- Each subagent must write out the corresponding findings file (`system_findings/` or `category_findings/`).
- Do not fabricate analysis results — all data must come from agent analysis output.
- Write the final report to `$RESULT_DIR/tracelens_output/baseline/analysis.md`.

After completion, the following artifacts are available under `$RESULT_DIR/tracelens_output/baseline/`:

- `analysis.md` — final composable report (system + compute sections)
- `category_data/<category>_findings.json` — per-category efficiency + `impact_estimates`
- `priority_data.json` — ranked bottlenecks driving the perf-improvement plot
- `metadata/model_info.json` — detected model architecture
- `perf_report.xlsx` + `perf_report_csvs/` — raw perf report
- `system_findings/` — CPU/idle + multi-kernel findings

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
- TraceLens CLI not installed: install path tried in order — `/hyperloom/TraceLens-internal`, `/opt/TraceLens`, `git clone $TRACELENS_GIT_URL /tmp/TraceLens-internal` — then `pip install -e <dir>` (see Step 5 install block).
- TraceLens orchestrator run fails (perf report, prepare, or any subagent): fall back to manual kernel analysis (Step 3); GEAK candidates from Step 6 still derive from raw trace parsing.
