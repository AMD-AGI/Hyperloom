# Action: Environment Setup

## Inputs
- User-specified CONFIG_YAML, NUM_GPUS, FRAMEWORK (optional — auto-detect if not provided)

## Procedure

### Step 1: Auto-detect environment

```bash
GPU_COUNT=$(python3 -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo 8)
GPU_TYPE=$(rocm-smi --showproductname 2>/dev/null | grep -o "MI[0-9]*[A-Za-z]*" | head -1 || echo "MI355X")

FRAMEWORK="${FRAMEWORK:-primus}"
if [ "$FRAMEWORK" = "primus" ]; then
    PRIMUS_ROOT="${PRIMUS_ROOT:-/workspace/Primus}"
    FRAMEWORK_VERSION=$(cd "$PRIMUS_ROOT" && git rev-parse --short HEAD 2>/dev/null || echo "unknown")
fi
```

### Step 2: Locate training config

```bash
# Auto-detect config if not provided
if [ -z "$CONFIG_YAML" ]; then
    CONFIG_YAML=$(find "$PRIMUS_ROOT/examples/megatron/configs" -name "*$GPU_TYPE*" -name "*.yaml" | head -1)
fi
```

User-specified values override auto-detected ones.

### Step 3: Set paths and env vars

```bash
SKILL_ROOT="${SKILL_ROOT:-.cursor/skills/training-optimization}"
SCRIPTS_DIR="$SKILL_ROOT/scripts"

export CONFIG_YAML="$CONFIG_YAML"
export NUM_GPUS="$GPU_COUNT"
export FRAMEWORK="$FRAMEWORK"
export PRIMUS_ROOT="$PRIMUS_ROOT"
export MASTER_PORT="${MASTER_PORT:-29500}"
```

### Step 4: Validate prerequisites

```bash
# Check Primus installation
[ -d "$PRIMUS_ROOT" ] || { echo "ERROR: Primus not found at $PRIMUS_ROOT"; exit 1; }

# Compile mock data helpers (required for MockGPTDataset)
make -C "$PRIMUS_ROOT/third_party/Megatron-LM/megatron/core/datasets" 2>/dev/null || true

# Kill any lingering training processes
pkill -9 -f "primus/cli/main.py" 2>/dev/null || true
sleep 3
```

### Step 5: Create results directory

```bash
TIMESTAMP=$(date +%Y-%m-%d-%H-%M)
RESULT_DIR="${RESULT_DIR:-/shared_nfs/training-optimization/results/${TIMESTAMP}}"
mkdir -p "$RESULT_DIR"
```

## Outputs
- All environment variables set
- `$SKILL_ROOT`, `$SCRIPTS_DIR`, `$PRIMUS_ROOT` paths validated
- `$RESULT_DIR` created
- Mock data C++ helpers compiled

## Failure Handling
- If no config found: ask user for CONFIG_YAML path
- If no GPUs detected: check ROCm installation
- If Primus not found: check `/workspace/Primus/` or user-specified path
