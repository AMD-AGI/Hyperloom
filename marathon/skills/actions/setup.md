# Action: Marathon Warm-Start

Marathon starts from an already-optimized baseline. This action ingests either a Sprint
handoff or a pre-optimized directory and sets up the Marathon state.

## Inputs
- **Mode A:** Sprint handoff directory (`$SPRINT_HANDOFF_DIR/handoff/`)
- **Mode B:** Pre-optimized baseline directory (contains launch script + optional patches/results)
- User-specified overrides for MODEL, TP, CONC, ISL, OSL (optional)

## Procedure

### Step 0: Detect warm-start mode

```bash
# Check for Sprint handoff first
if [ -d "$SPRINT_HANDOFF_DIR/handoff" ] && [ -f "$SPRINT_HANDOFF_DIR/handoff/config.json" ]; then
    WARMSTART_MODE="sprint"
    echo "=== WARM-START MODE A: Sprint Handoff ==="
elif [ -n "$BASELINE_DIR" ] && [ -d "$BASELINE_DIR" ]; then
    WARMSTART_MODE="baseline"
    echo "=== WARM-START MODE B: Pre-Optimized Baseline ==="
else
    WARMSTART_MODE="cold"
    echo "=== COLD START: No Sprint handoff or baseline found ==="
    echo "Running as standalone Marathon — will profile from scratch"
fi
```

### Step 1 (Mode A): Ingest Sprint handoff

```bash
# Read Sprint config
python3 -c "
import json
config = json.load(open('$SPRINT_HANDOFF_DIR/handoff/config.json'))
print(f'Model: {config[\"model_name\"]}')
print(f'Framework: {config[\"framework\"]} v{config.get(\"framework_version\", \"unknown\")}')
print(f'GPU: {config[\"gpu_count\"]}x {config[\"gpu_type\"]}')
print(f'TP: {config[\"tp\"]}')
print(f'Sprint throughput: {config[\"optimized_tput_per_gpu\"]:.2f} tok/s/GPU')
print(f'Sprint gain: {config[\"cumulative_gain_pct\"]:.1f}%')
if config.get('target_tput_per_gpu'):
    print(f'Target: {config[\"target_tput_per_gpu\"]:.2f} tok/s/GPU (gap: {config[\"target_gap_pct\"]:.1f}%)')
"

# Apply Sprint patches
if [ -d "$SPRINT_HANDOFF_DIR/handoff/patches" ]; then
    for patch in "$SPRINT_HANDOFF_DIR/handoff/patches"/*.patch; do
        echo "Applying Sprint patch: $(basename $patch)"
        # Determine target repo from patch header
        git apply --check "$patch" 2>/dev/null && git apply "$patch"
    done
fi

# Load Marathon opportunities
python3 -c "
import json
opps = json.load(open('$SPRINT_HANDOFF_DIR/handoff/opportunities.json'))
print(f'Marathon opportunities: {len(opps)}')
for opp in opps:
    print(f'  - [{opp[\"type\"]}] {opp[\"kernel_name\"]} ({opp[\"gpu_pct\"]:.1f}% GPU) — {opp.get(\"recommended_marathon_action\", \"N/A\")}')
    print(f'    Tags: {opp.get(\"tags\", [])}')
"

# Load profile summary
python3 -c "
import json
profile = json.load(open('$SPRINT_HANDOFF_DIR/handoff/profile_summary.json'))
print('Tier breakdown from Sprint:')
for tier, pct in profile.get('tier_breakdown', {}).items():
    print(f'  {tier}: {pct:.1f}%')
"
```

### Step 1 (Mode B): Ingest pre-optimized baseline

```bash
# Find launch script
LAUNCH_SCRIPT=$(find "$BASELINE_DIR" -name "launch_server*" -o -name "run_server*" | head -1)
if [ -z "$LAUNCH_SCRIPT" ]; then
    echo "ERROR: No launch script found in $BASELINE_DIR"
    exit 1
fi

echo "Found launch script: $LAUNCH_SCRIPT"

# Extract config from launch script
python3 -c "
import re, json, sys

content = open('$LAUNCH_SCRIPT').read()

# Extract model path
model_match = re.search(r'--model[- ]path\s+(\S+)|vllm\s+serve\s+(\S+)', content)
model_path = (model_match.group(1) or model_match.group(2)) if model_match else 'UNKNOWN'

# Extract all --flag value pairs
flags = re.findall(r'--([\w-]+)\s+(\S+)', content)
flag_dict = {k: v for k, v in flags}

# Extract env vars
envs = re.findall(r'export\s+(\w+)=(\S+)|(\w+)=(\S+)\s+\\\\', content)

print(f'Model: {model_path}')
print(f'Flags: {json.dumps(flag_dict, indent=2)}')
"

# Apply any patches in the baseline directory
if [ -d "$BASELINE_DIR/patches" ]; then
    for patch in "$BASELINE_DIR/patches"/*.patch; do
        echo "Applying baseline patch: $(basename $patch)"
        git apply --check "$patch" 2>/dev/null && git apply "$patch"
    done
fi
```

### Step 1 (Cold start): Basic setup

```bash
# Standard environment detection
GPU_COUNT=$(nvidia-smi -L 2>/dev/null | wc -l || rocm-smi --showproductname 2>/dev/null | grep -c "GPU")
GPU_TYPE=$(rocm-smi --showproductname 2>/dev/null | head -1 || nvidia-smi --query-gpu=name --format=csv,noheader | head -1)

echo "GPUs: ${GPU_COUNT}x ${GPU_TYPE}"
echo "Cold start — will run baseline + profile before Marathon DFS"
```

### Step 2: Initialize Marathon state

```python
import json

state = {
    "model_name": MODEL_NAME,
    "model_class": MODEL_CLASS,
    "framework": FRAMEWORK,
    "tp": TP,
    "gpu_type": GPU_TYPE,
    "gpu_count": GPU_COUNT,
    "num_nodes": NUM_NODES,

    "sprint_tput_per_gpu": SPRINT_TPUT,  # 0 if cold start
    "baseline_tput_per_gpu": SPRINT_TPUT,
    "current_tput_per_gpu": SPRINT_TPUT,
    "cumulative_gain_pct": 0.0,  # Marathon gain starts at 0

    "target_tput_per_gpu": TARGET_TPUT,
    "target_gap_pct": TARGET_GAP,

    "kernel_dispatch_map": {},
    "untuned_shapes": [],
    "dispatch_bugs_found": [],

    "baseline_accuracy": None,
    "accuracy_threshold": 0.01,

    "action_stack": [],
    "completed_actions": [],
    "kernel_candidates": [],

    "pending_kernel_tasks": [],
    "kernel_results": {},

    "current_time_tier": "tier1",
    "checkpoint_path": None,
    "dream_count": 0,
    "last_dream_ts": None,
    "crash_count": 0,
    "crash_log": [],
    "strategies_tested": [],
    "tier_breakdown": {},
    "loop_signatures": [],

    "total_wall_minutes": 0,
    "total_kernel_opt_submissions": 0,
    "consecutive_discards": 0,
    "backend_wins": {},
    "frameworks_rebuilt": [],
}

json.dump(state, open(f"{RESULT_DIR}/state.json", "w"), indent=2)
```

### Step 3: Pre-populate action stack from opportunities

If Sprint handoff is available, convert `opportunities.json` into scored Marathon actions:

```python
for opp in opportunities:
    base_score = {
        'deep-kernel-analysis': 9,
        'operator-tuning': 7,
        'comm-optimization': 5,
        'deep-kernel-opt': 6,
    }.get(opp['type'], 5)

    # Boost from tags
    tag_boost = 0
    if 'register-pressure-fixable' in opp.get('tags', []):
        tag_boost += 3
    if 'shape-tuning-untested' in opp.get('tags', []):
        tag_boost += 2
    if 'oob-untested' in opp.get('tags', []):
        tag_boost += 2
    if 'marathon-candidate' in opp.get('tags', []):
        tag_boost += 3

    score = base_score + tag_boost
    state['action_stack'].append((score, opp['type'], opp))
```

## Outputs
- `$RESULT_DIR/state.json` — initialized Marathon state
- Applied patches from Sprint or baseline
- Pre-populated action stack (if Sprint handoff available)
- Environment verified and ready for re-profile

## Next Step
Proceed to **Step 1: RE-PROFILE** (`actions/profile.md`) to get a fresh trace
on the optimized baseline before starting Marathon DFS.
