# Action: Environment Setup

## Inputs
- `$REPO_ROOT` (required) — path to the user's GPU repo
- `$GPU` (optional) — GPU type override (e.g. MI300X / MI355X)
- `$TIME_BUDGET_MIN` (optional, default 90)

## Procedure

### Step 1: Validate repo exists
```bash
: "${REPO_ROOT:?REPO_ROOT required}"
[ -d "$REPO_ROOT" ] || { echo "ERROR: $REPO_ROOT does not exist"; exit 1; }
```

### Step 2: Detect GPU + ROCm
```bash
GPU_NAME=$(rocm-smi --showproductname 2>/dev/null | grep -oE "MI[0-9]+[A-Z]*" | head -1)
GPU_ARCH=$(rocminfo 2>/dev/null | grep -m1 -oE "gfx[0-9]+[a-z]*")
ROCM_VERSION=$(cat /opt/rocm/.info/version 2>/dev/null || hipconfig --version 2>/dev/null)
GPU_COUNT=$(rocm-smi --showid 2>/dev/null | grep -oE "GPU\[[0-9]+\]" | sort -u | wc -l)

export GPU="${GPU:-$GPU_NAME}"
export GPU_ARCH GPU_COUNT ROCM_VERSION

[ -n "$GPU_ARCH" ] || { echo "ERROR: no AMD GPU detected (rocminfo failed)"; exit 1; }
echo "GPU: $GPU ($GPU_ARCH), $GPU_COUNT devices, ROCm $ROCM_VERSION"
```

### Step 3: Create result directory
```bash
TIMESTAMP=$(date +%Y-%m-%d-%H-%M)
export RESULT_DIR="${RESULT_DIR:-/tmp/hyperloom-results/$(basename "$REPO_ROOT")-$TIMESTAMP}"
mkdir -p "$RESULT_DIR/patches" "$RESULT_DIR/profiles"
echo "Results: $RESULT_DIR"
```

### Step 4: Snapshot baseline git state
```bash
cd "$REPO_ROOT"
if git rev-parse --git-dir >/dev/null 2>&1; then
    BASELINE_SHA=$(git rev-parse HEAD)
    git status --porcelain > "$RESULT_DIR/git_status_baseline.txt"
    [ -s "$RESULT_DIR/git_status_baseline.txt" ] && \
        echo "WARNING: working tree not clean — patches may apply on top of local changes"
    echo "Baseline SHA: $BASELINE_SHA"
    export BASELINE_SHA
fi
```

### Step 5: Initialize state file
```bash
cat > "$RESULT_DIR/state.env" <<STATE
REPO_ROOT=$REPO_ROOT
RESULT_DIR=$RESULT_DIR
GPU=$GPU
GPU_ARCH=$GPU_ARCH
GPU_COUNT=$GPU_COUNT
ROCM_VERSION=$ROCM_VERSION
BASELINE_SHA=${BASELINE_SHA:-}
TIME_BUDGET_MIN=${TIME_BUDGET_MIN:-90}
STATE
```

## Outputs
- `$RESULT_DIR` populated
- `$RESULT_DIR/state.env` — sourceable across all subsequent actions
- GPU info exported

## Failure Handling
- No AMD GPU: stop with clear error
- No git: continue, but warn that patch tracking degrades
