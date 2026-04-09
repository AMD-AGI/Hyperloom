# Action: Baseline Training Run

## Inputs
- `$CONFIG_YAML`, `$NUM_GPUS`, `$PRIMUS_ROOT`, `$MASTER_PORT`
- `$RESULT_DIR` for storing outputs

## Procedure

### Step 1: Kill stale processes

```bash
pkill -9 -f "primus/cli/main.py" 2>/dev/null || true
sleep 5
```

### Step 2: Run baseline training

```bash
cd "$PRIMUS_ROOT"

torchrun --nproc_per_node=$NUM_GPUS --master_port=$MASTER_PORT \
  -m primus.cli.main train pretrain \
  --config "$CONFIG_YAML" \
  profile=false use_pytorch_profiler=false \
  2>&1 | tee "$RESULT_DIR/baseline.log"
```

### Step 3: Extract ms/iter from log

```python
import re

with open(f"{RESULT_DIR}/baseline.log") as f:
    lines = f.readlines()

# Look for iteration timing lines (format varies by framework)
# Primus/Megatron: "elapsed time per iteration (ms): 13265.3"
# Alternative: "iteration N / M | ... | time (ms): XXXX"
iter_times = []
for line in lines:
    m = re.search(r"elapsed time per iteration \(ms\):\s*([\d.]+)", line)
    if not m:
        m = re.search(r"time \(ms\):\s*([\d.]+)", line)
    if not m:
        m = re.search(r"iter_time.*?(\d+\.?\d*)\s*ms", line, re.IGNORECASE)
    if m:
        iter_times.append(float(m.group(1)))

# Use iterations 6–10 (skip warmup 1–5)
if len(iter_times) >= 10:
    baseline_ms = sum(iter_times[5:10]) / 5
elif len(iter_times) >= 6:
    baseline_ms = sum(iter_times[5:]) / len(iter_times[5:])
else:
    baseline_ms = sum(iter_times) / len(iter_times)

print(f"Baseline ms/iter: {baseline_ms:.1f}")
```

### Step 4: Verify GBS from log

```python
gbs_match = None
for line in lines:
    m = re.search(r"global.batch.size.*?(\d+)", line, re.IGNORECASE)
    if m:
        gbs_match = int(m.group(1))

assert gbs_match is not None, "Could not find GBS in training log"
baseline_gbs = gbs_match
print(f"GBS: {baseline_gbs}")
```

### Step 5: Initialize results log

```
cat > "$RESULT_DIR/results.tsv" <<EOF
attempt	ms_per_iter	speedup_pct	status	description
0	${BASELINE_MS}	0.0	baseline	Baseline (${NUM_GPUS} GPU, config: $(basename $CONFIG_YAML))
EOF
```

### Step 6: Capture accuracy reference (optional)

For convergence tracking, record the loss values from the baseline:

```python
losses = []
for line in lines:
    m = re.search(r"lm loss.*?:\s*([\d.]+)", line, re.IGNORECASE)
    if m:
        losses.append(float(m.group(1)))

import json
with open(f"{RESULT_DIR}/accuracy_reference.json", "w") as f:
    json.dump({"baseline_losses": losses, "gbs": baseline_gbs}, f, indent=2)
```

## Outputs
- `baseline_ms_per_iter`: average ms/iter from iterations 6–10
- `baseline_gbs`: verified global batch size
- `$RESULT_DIR/baseline.log`: full training log
- `$RESULT_DIR/results.tsv`: initialized results tracking
- `$RESULT_DIR/accuracy_reference.json`: loss values for convergence tracking

## Failure Handling
- If training crashes: check for port conflicts (increment `MASTER_PORT`), OOM, or missing deps
- If ms/iter not found in log: check log format, try alternative regex patterns
- If GBS not found: parse config YAML directly
