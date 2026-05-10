# Action: Environment Setup

## Inputs
- User-specified MODEL, TP, CONC, ISL, OSL, FRAMEWORK (optional — auto-detect if not provided)

## Procedure

### Step 0 (CRITICAL): Set PATH; bootstrap only inside RayJob

**ALWAYS prepend `/opt/venv/bin` to PATH before any python3 command.** The system
`/usr/bin/python3` does NOT have sglang/vllm/numpy installed. Every bash command
in this skill MUST start with:

```bash
export PATH="/opt/venv/bin:$PATH"
```

Do **not** run `scripts/bootstrap.sh` from the Claw sandbox or any external client.
Bootstrap runs only inside the RayJob after it is created (see "Remote mode" below).
Once a RayJob command has sourced `/etc/profile.d/hyperloom-env.sh`, it can reuse the
bootstrap-generated environment for later commands in that RayJob.

### Step 1: Auto-detect environment

```bash
export PATH="/opt/venv/bin:$PATH"

MODEL="${MODEL:-$(ls -d /shared_nfs/*/models/*/ 2>/dev/null | head -1)}"
GPU_COUNT="${GPU_COUNT:-$(amd-smi list 2>/dev/null | grep "^GPU:" | wc -l)}"
GPU_TYPE="${GPU_TYPE:-$(rocm-smi --showproductname 2>/dev/null | grep "GFX Version" | head -1 | grep -o "gfx[0-9]*")}"
INFERENCEX_PATH="${INFERENCEX_PATH:-$(ls -d /shared_nfs/*/InferenceX 2>/dev/null | head -1)}"

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

MODE="remote"
WORKSPACE_ROOT="${WORKSPACE_ROOT:-/shared_nfs/inference-optimization}"
OOB_RAY_CLI="python3 $SCRIPTS_DIR/oob_ray_submit.py"
OOB_CLI="${OOB_CLI:-/opt/venv/bin/oob}"

# Enables exec_on_gpu for Ray Dashboard REST dispatch.
source "$SCRIPTS_DIR/executor.sh"

export MODE="$MODE"
export WORKSPACE_ROOT="$WORKSPACE_ROOT"
export MODEL="$MODEL"
export TP="$TP"
export CONC="$CONC"
export ISL="${ISL:-1024}"
export OSL="${OSL:-256}"
export FRAMEWORK="${FRAMEWORK:-sglang}"
export INFERENCEX_PATH="$INFERENCEX_PATH"
[ -n "${OOB_RAY_CLI:-}" ] && export OOB_RAY_CLI
[ -n "${OOB_CLI:-}" ] && export OOB_CLI
```

`run_baseline.sh` writes `run_context.env` into `$RESULT_DIR`. Reuse it for subsequent steps.

## Outputs
- All environment variables set
- `$SKILL_ROOT`, `$SCRIPTS_DIR` paths validated
- `$RESULT_DIR` created

**Remote mode:** After environment setup, create the RayJob before proceeding. See
[`../modes/REMOTE.md`](../modes/REMOTE.md) "RayJob Lifecycle" for the full `workload_create`
payloads, wait logic, and remote execution environment setup.

After the RayJob is running and `HEAD_IP` or `RAY_DASHBOARD_URL` is set, run the BYOI bootstrap
and CLI preflight inside the RayJob before benchmark/profile/kernel-opt:

```bash
exec_on_gpu "
export PATH='/opt/venv/bin:'\"\$PATH\"
export MODE=remote
export SKILL_ROOT='$SKILL_ROOT'
export HYPERLOOM_BUNDLE='\${HYPERLOOM_BUNDLE:-/wekafs/fully-local}'
if [ ! -f /opt/hyperloom/.bootstrap_done ] && [ -d \"\$HYPERLOOM_BUNDLE\" ]; then
  bash '$SCRIPTS_DIR/bootstrap.sh'
fi
[ -f /etc/profile.d/hyperloom-env.sh ] && . /etc/profile.d/hyperloom-env.sh
export MODE=remote
command -v oob || echo 'WARN: oob CLI missing'
command -v codex || command -v claude || echo 'WARN: codex/claude CLI missing'
command -v TraceLens_generate_perf_report_pytorch_inference || echo 'WARN: TraceLens CLI missing'
"
```


## Failure Handling
- If no model found: ask user for MODEL path
- If no GPUs detected: check `amd-smi` / `rocm-smi` installation
- If InferenceX not found: check `/shared_nfs/*/InferenceX/`
