# Action: Environment Setup

## Inputs
- User-specified config shell script path (default: `config_MI355X_1x8x1_fp8.sh`)
- MLPerf code directory (default: `/root/Hyperloom-plus-mlperf/training_optimization/mlperf`)

## Procedure

### Step 1: Auto-detect environment

```bash
GPU_COUNT=$(python3 -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo 8)
GPU_TYPE=$(rocm-smi --showproductname 2>/dev/null | grep -o "MI[0-9]*[A-Za-z]*" | head -1 || echo "MI355X")
```

### Step 2: Source config and set paths

```bash
MLPERF_DIR="${MLPERF_DIR:-/root/Hyperloom-plus-mlperf/training_optimization/mlperf}"
CONFIG_SH="${CONFIG_SH:-$MLPERF_DIR/config_MI355X_1x8x1_fp8.sh}"

cd "$MLPERF_DIR"
source "$CONFIG_SH"
```

### Step 3: Create necessary directories

```bash
mkdir -p "$LOGDIR" "$(dirname $MLLOG_OUTPUT_FILE)" /root/mlperf_primus/conf
cp "$MLPERF_DIR/conf/gpt_oss_20B-pretrain-fp8.yaml" /root/mlperf_primus/conf/
```

### Step 4: Setup container symlinks

```bash
bash "$MLPERF_DIR/setup_container_symlinks.sh"
```

This creates:
- `/workspace/code` → `$MLPERF_DIR`
- `/data` → `$DATADIR`
- `/model` → `$MODELDIR`
- `/results` → `$LOGDIR`

### Step 4.5: Validate trial infrastructure

```bash
SKILL_ROOT="${SKILL_ROOT:-.cursor/skills/mlperf-optimization}"

# Verify trial_monitor.py
[ -f "$SKILL_ROOT/scripts/trial_monitor.py" ] || { echo "ERROR: trial_monitor.py not found"; exit 1; }
python3 "$SKILL_ROOT/scripts/trial_monitor.py" --help >/dev/null 2>&1 || { echo "ERROR: trial_monitor.py not executable"; exit 1; }

# Verify quiet config functions
source "$SKILL_ROOT/scripts/apply_quiet_config.sh"
quiet_yaml "$EXP" && restore_yaml "$EXP" || { echo "ERROR: quiet_yaml/restore_yaml failed"; exit 1; }

# Verify common.sh loads
source "$SKILL_ROOT/scripts/common.sh" || { echo "ERROR: common.sh failed to source"; exit 1; }
```

### Step 5: Validate prerequisites

```bash
# Check data exists
[ -f /data/c4-train.en_6_text_document.bin ] || { echo "ERROR: Training data not found"; exit 1; }
[ -f /data/c4-validation-91205-samples.en_text_document.bin ] || { echo "ERROR: Validation data not found"; exit 1; }

# Check Primus
[ -d "$PRIMUS_PATH" ] || { echo "ERROR: Primus not found at $PRIMUS_PATH"; exit 1; }

# Check config
[ -f "$EXP" ] || { echo "ERROR: Config not found at $EXP"; exit 1; }

# Kill any lingering training processes
source "$SKILL_ROOT/scripts/common.sh"
kill_training
```

### Step 6: Apply runtime tunables (optional)

```bash
# Only if running on bare metal or have sudo access
if [ -w /proc/sys/vm/drop_caches ]; then
    bash "$MLPERF_DIR/runtime_tunables.sh"
fi
```

### Step 7: Create results directory

```bash
TIMESTAMP=$(date +%Y-%m-%d-%H-%M)
SKILL_ROOT="${SKILL_ROOT:-.cursor/skills/mlperf-optimization}"
RESULT_DIR="${RESULT_DIR:-/root/mlperf_results/${TIMESTAMP}}"
mkdir -p "$RESULT_DIR"
```

## Outputs
- All environment variables set (from config_MI355X_1x8x1_fp8.sh)
- Symlinks verified
- `$RESULT_DIR` created
- `$SKILL_ROOT`, `$MLPERF_DIR` paths validated
- `trial_monitor.py` validated and executable
- `quiet_yaml`/`restore_yaml` functions verified
- `common.sh` sourced successfully (run_mlperf_trial available)

## Failure Handling
- If data not found: check `$DATADIR` path
- If no GPUs detected: check ROCm installation
- If Primus not found: check `/workspace/Primus/`
- If symlink setup fails: create directories manually
