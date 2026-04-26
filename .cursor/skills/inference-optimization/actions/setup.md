# Action: Environment Setup

## Inputs
- User-specified MODEL, TP, CONC, ISL, OSL, FRAMEWORK (optional — auto-detect if not provided)

## Procedure

### Step 0 (CRITICAL): Set PATH and bootstrap BYOI

**ALWAYS prepend `/opt/venv/bin` to PATH before any python3 command.** The system
`/usr/bin/python3` does NOT have sglang/vllm/numpy installed. Every bash command
in this skill MUST start with:

```bash
export PATH="/opt/venv/bin:$PATH"

SKILL_ROOT="${SKILL_ROOT:-.cursor/skills/inference-optimization}"
BOOTSTRAP_SCRIPT="$SKILL_ROOT/scripts/bootstrap.sh"
HYPERLOOM_BUNDLE="${HYPERLOOM_BUNDLE:-/wekafs/fully-local}"

if [ "${MODE:-}" = "fully-local" ] \
   && [ ! -f /opt/entrypoint.sh ] \
   && [ ! -f /opt/hyperloom/.bootstrap_done ] \
   && [ -d "$HYPERLOOM_BUNDLE" ]; then
    bash "$BOOTSTRAP_SCRIPT"
    . /etc/profile.d/hyperloom-env.sh
elif [ -f /opt/hyperloom/.bootstrap_done ] && [ -f /etc/profile.d/hyperloom-env.sh ]; then
    . /etc/profile.d/hyperloom-env.sh
fi
```

### Step 1: Auto-detect environment

```bash
export PATH="/opt/venv/bin:$PATH"

MODEL="${MODEL:-$(ls -d /shared_nfs/*/models/*/ 2>/dev/null | head -1)}"
GPU_COUNT="${GPU_COUNT:-$(amd-smi list 2>/dev/null | grep "^GPU:" | wc -l)}"
GPU_TYPE="${GPU_TYPE:-$(rocm-smi --showproductname 2>/dev/null | grep "GFX Version" | head -1 | grep -o "gfx[0-9]*")}"
INFERENCEX_PATH="${INFERENCEX_PATH:-$(ls -d /shared_nfs/*/InferenceX 2>/dev/null | head -1)}"
if [ "${MODE:-}" = "fully-local" ]; then
    INFERENCEX_PATH="${INFERENCEX_PATH:-/opt/hyperloom/InferenceX}"
fi

FRAMEWORK="${FRAMEWORK:-sglang}"
if [ "$FRAMEWORK" = "vllm" ]; then
    FRAMEWORK_VERSION=$(python3 -c "import vllm; print(vllm.__version__)" 2>/dev/null)
else
    FRAMEWORK_VERSION=$(python3 -c "import sglang; print(sglang.__version__)" 2>/dev/null)
fi

TP="${TP:-$GPU_COUNT}"
if [ -z "${CONC:-}" ]; then
    if [ "$TP" -le 1 ]; then CONC=4; elif [ "$TP" -le 4 ]; then CONC=32; else CONC=64; fi
fi
```

User-specified values override auto-detected ones. **NEVER override user-specified TP**
— if the prompt says TP=8, use TP=8 even if GPU_COUNT differs.

### Step 2: Set paths and env vars

```bash
SKILL_ROOT="${SKILL_ROOT:-.cursor/skills/inference-optimization}"
SCRIPTS_DIR="$SKILL_ROOT/scripts"

# Mode detection
if [ "${MODE:-}" = "fully-local" ]; then
    MODE="fully-local"
    WORKSPACE_ROOT="${WORKSPACE_ROOT:-/opt/hyperloom}"
    GEAK_CLI="python3 $SCRIPTS_DIR/geak_ray_submit.py"
    OOB_CLI="${OOB_CLI:-oob}"
elif [ "${MODE:-}" = "claw" ]; then
    MODE="claw"
    WORKSPACE_ROOT="${WORKSPACE_ROOT:-/shared_nfs/inference-optimization}"
elif [ "${GEAK_LOCAL:-true}" = "true" ]; then
    MODE="local"
    WORKSPACE_ROOT="${WORKSPACE_ROOT:-/workspace/inference-optimization}"
else
    MODE="claw"
    WORKSPACE_ROOT="${WORKSPACE_ROOT:-/shared_nfs/inference-optimization}"
fi

# Source mode-specific helpers
if [ "$MODE" = "fully-local" ]; then
    source "$SCRIPTS_DIR/common.sh"
    WORKSPACE_ROOT="/opt/hyperloom"
else
    # Enables exec_on_gpu for local/claw dispatch.
    source "$SCRIPTS_DIR/executor.sh"
fi

export MODE="$MODE"
export WORKSPACE_ROOT="$WORKSPACE_ROOT"
export MODEL="$MODEL"
export TP="$TP"
export CONC="$CONC"
export ISL="${ISL:-1024}"
export OSL="${OSL:-256}"
export FRAMEWORK="${FRAMEWORK:-sglang}"
export INFERENCEX_PATH="$INFERENCEX_PATH"
[ -n "${GEAK_CLI:-}" ] && export GEAK_CLI
[ -n "${OOB_CLI:-}" ] && export OOB_CLI
```

`run_baseline.sh` writes `run_context.env` into `$RESULT_DIR`. Reuse it for subsequent steps.

## Outputs
- All environment variables set
- `$SKILL_ROOT`, `$SCRIPTS_DIR` paths validated
- `$RESULT_DIR` created

**Claw mode:** After environment setup, create the RayJob before proceeding. See
[`../modes/CLAW.md`](../modes/CLAW.md) "RayJob Lifecycle" for the full `workload_create`
payloads, wait logic, and claw execution environment setup.

## Failure Handling
- If no model found: ask user for MODEL path
- If no GPUs detected: check `amd-smi` / `rocm-smi` installation
- If InferenceX not found: check `/shared_nfs/*/InferenceX/`
