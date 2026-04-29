#!/usr/bin/env bash
# =============================================================================
# inference-optimizer/scripts/preflight.sh
#
# Single-command pre-flight check for a real-GPU inference-optimizer run.
# Invoked by the Cursor agent BEFORE launching ``python -m
# inference_optimizer`` so we catch the 11 known classes of "won't even
# start" failure with copy-pasteable fixes.
#
# Usage:
#     bash src/inference_optimizer/scripts/preflight.sh
#     # or with explicit env:
#     MODEL_PATH=/data/x INFERENCEX_PATH=/opt/InferenceX bash .../preflight.sh
#
# Exit codes:
#     0  all checks passed (run is safe to launch)
#     1  at least one hard requirement failed; STDOUT/STDERR explains
#        which one and how to fix it
#
# Env it consumes (all optional except MODEL_PATH):
#     MODEL_PATH                 (required) path to model weights
#     INFERENCEX_PATH            path to InferenceX checkout
#     ANTHROPIC_API_KEY / OPENAI_API_KEY  at least one
#     ANTHROPIC_BASE_URL         (corp proxy → triggers TLS hint)
#     INFERENCE_OPTIMIZER_SESSION_ROOT  defaults to /tmp/io-sessions
#     PY                         python binary (default: python3)
# =============================================================================
set -uo pipefail

PY="${PY:-python3}"
SESSION_ROOT="${INFERENCE_OPTIMIZER_SESSION_ROOT:-/tmp/io-sessions}"

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

OK_COUNT=0
WARN_COUNT=0
FAIL_COUNT=0
HINTS=()

ok()    { printf "${GREEN}[ok]${NC}    %s\n" "$1"; OK_COUNT=$((OK_COUNT+1)); }
warn()  { printf "${YELLOW}[warn]${NC}  %s\n" "$1"; WARN_COUNT=$((WARN_COUNT+1)); }
fail()  { printf "${RED}[fail]${NC}  %s\n" "$1" >&2; FAIL_COUNT=$((FAIL_COUNT+1)); }
hint()  { HINTS+=("$1"); }

echo "============================================================"
echo "Inference Optimizer pre-flight"
echo "Date:           $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "Host:           $(hostname)"
echo "Python:         ${PY} ($(${PY} --version 2>&1))"
echo "Session root:   ${SESSION_ROOT}"
echo "============================================================"

# ---------------------------------------------------------------------------
# 1. Python interpreter usable
# ---------------------------------------------------------------------------
if ${PY} -c 'import sys; assert sys.version_info >= (3, 10), sys.version' 2>/dev/null; then
    ok "1. python interpreter is >=3.10 ($(${PY} --version 2>&1))"
else
    fail "1. python interpreter ${PY} is missing or <3.10"
    hint "  Install Python 3.10+ or set PY=/path/to/python3.10"
fi

# ---------------------------------------------------------------------------
# 2. Required Python libs (sglang | vllm) + openai + claude_agent_sdk
# ---------------------------------------------------------------------------
if ${PY} - <<'PYEOF' 2>/dev/null
import importlib, sys
missing = []
for mod in ("openai", "yaml", "claude_agent_sdk"):
    try:
        importlib.import_module(mod)
    except Exception:
        missing.append(mod)
sg = vl = None
try:
    sg = importlib.import_module("sglang")
except Exception:
    pass
try:
    vl = importlib.import_module("vllm")
except Exception:
    pass
if not (sg or vl):
    missing.append("sglang|vllm (need at least one)")
if missing:
    print(",".join(missing))
    sys.exit(1)
print(f"sglang={getattr(sg, '__version__', None)} vllm={getattr(vl, '__version__', None)}")
PYEOF
then
    ok "2. required python libs OK"
else
    fail "2. required python libs missing"
    hint "  pip install -r src/inference_optimizer/requirements.txt"
    hint "  (and at least one of: pip install sglang  OR  pip install vllm)"
fi

# ---------------------------------------------------------------------------
# 3. torch importable + CUDA/ROCm available
# ---------------------------------------------------------------------------
TORCH_INFO=$(${PY} - <<'PYEOF' 2>/dev/null
import torch
ok = torch.cuda.is_available()
n  = torch.cuda.device_count() if ok else 0
print(f"torch={torch.__version__} cuda_available={ok} devices={n}")
PYEOF
)
if [ -n "$TORCH_INFO" ]; then
    if echo "$TORCH_INFO" | grep -q "cuda_available=True"; then
        ok "3. torch + GPU runtime: $TORCH_INFO"
    else
        warn "3. torch installed but no CUDA/ROCm device visible: $TORCH_INFO"
        hint "  Check ROCR_VISIBLE_DEVICES / HIP_VISIBLE_DEVICES / CUDA_VISIBLE_DEVICES"
    fi
else
    fail "3. torch not importable from ${PY}"
    hint "  This usually means sglang/vllm picked a different python."
    hint "  Confirm: which ${PY}; ${PY} -c 'import torch'"
fi

# ---------------------------------------------------------------------------
# 4. GPU SMI on PATH
# ---------------------------------------------------------------------------
SMI=""
for tool in amd-smi rocm-smi nvidia-smi; do
    if command -v $tool >/dev/null 2>&1; then
        SMI="$tool"
        break
    fi
done
if [ -n "$SMI" ]; then
    ok "4. GPU SMI on PATH: $SMI"
else
    warn "4. no GPU SMI found (amd-smi / rocm-smi / nvidia-smi)"
    hint "  Install ROCm or CUDA tools, or set GPU_COUNT=N to bypass auto-probe"
fi

# ---------------------------------------------------------------------------
# 5. GPU count > 0
# ---------------------------------------------------------------------------
GPU_N=0
if [ "$SMI" = "amd-smi" ]; then
    GPU_N=$(amd-smi list 2>/dev/null | grep -c '^GPU:' || echo 0)
elif [ "$SMI" = "rocm-smi" ]; then
    GPU_N=$(rocm-smi --showid 2>/dev/null | grep -c '^GPU\[' || echo 0)
    GPU_N=$((GPU_N / 5))   # 5 ID lines per GPU
elif [ "$SMI" = "nvidia-smi" ]; then
    GPU_N=$(nvidia-smi -L 2>/dev/null | wc -l || echo 0)
fi
if [ "$GPU_N" -gt 0 ] 2>/dev/null; then
    ok "5. detected $GPU_N GPU(s)"
else
    warn "5. could not auto-detect GPU count via $SMI"
    hint "  Set GPU_COUNT=N explicitly via env"
fi

# ---------------------------------------------------------------------------
# 6. MODEL_PATH points to readable directory with safetensors
# ---------------------------------------------------------------------------
MODEL_PATH="${MODEL_PATH:-}"
if [ -z "$MODEL_PATH" ]; then
    fail "6. MODEL_PATH env var unset"
    hint "  export MODEL_PATH=/path/to/Qwen3-30B-A3B   # or HF repo id"
elif [ ! -d "$MODEL_PATH" ] && [ ! -f "$MODEL_PATH" ]; then
    # Treat as HF repo id when not on disk
    case "$MODEL_PATH" in
        */*) ok "6. MODEL_PATH looks like an HF repo id ($MODEL_PATH) — sglang will download" ;;
        *)   fail "6. MODEL_PATH=$MODEL_PATH neither exists nor looks like an HF repo id" ;;
    esac
else
    SHARDS=$(ls "$MODEL_PATH"/*.safetensors 2>/dev/null | wc -l || echo 0)
    if [ "$SHARDS" -gt 0 ] 2>/dev/null; then
        ok "6. MODEL_PATH=$MODEL_PATH ($SHARDS safetensors shards)"
    elif [ -f "$MODEL_PATH/config.json" ]; then
        warn "6. MODEL_PATH=$MODEL_PATH has config.json but no .safetensors"
        hint "  Check for .bin or .pt weights; may still work"
    else
        fail "6. MODEL_PATH=$MODEL_PATH has no config.json or .safetensors"
    fi
fi

# ---------------------------------------------------------------------------
# 7. INFERENCEX_PATH/benchmarks/benchmark_lib.sh exists
# ---------------------------------------------------------------------------
INFERENCEX_PATH="${INFERENCEX_PATH:-}"
if [ -z "$INFERENCEX_PATH" ]; then
    warn "7. INFERENCEX_PATH not set — executors will fall back to LLM-only path"
    hint "  export INFERENCEX_PATH=/hyperloom/InferenceX  (or wherever the checkout is)"
elif [ -f "$INFERENCEX_PATH/benchmarks/benchmark_lib.sh" ]; then
    ok "7. INFERENCEX_PATH=$INFERENCEX_PATH (benchmark_lib.sh present)"
else
    fail "7. INFERENCEX_PATH=$INFERENCEX_PATH missing benchmarks/benchmark_lib.sh"
    hint "  git clone https://github.com/AMD-AIG-AIMA/InferenceX  $INFERENCEX_PATH"
fi

# ---------------------------------------------------------------------------
# 8. At least one LLM backend key set (ANTHROPIC_API_KEY or OPENAI_API_KEY)
# ---------------------------------------------------------------------------
HAS_ANTHROPIC=$(test -n "${ANTHROPIC_API_KEY:-}${ANTHROPIC_AUTH_TOKEN:-}" && echo yes || echo no)
HAS_OPENAI=$(test -n "${OPENAI_API_KEY:-}" && echo yes || echo no)
if [ "$HAS_ANTHROPIC" = "yes" ] || [ "$HAS_OPENAI" = "yes" ]; then
    ok "8. backend key set: anthropic=$HAS_ANTHROPIC openai=$HAS_OPENAI"
else
    fail "8. neither ANTHROPIC_API_KEY nor OPENAI_API_KEY is set"
    hint "  export ANTHROPIC_API_KEY=ak-...     # for --backend claude"
    hint "  export OPENAI_API_KEY=sk-...        # for --backend codex"
fi

# ---------------------------------------------------------------------------
# 9. Corp-proxy / self-signed-cert guidance
# ---------------------------------------------------------------------------
if [ -n "${ANTHROPIC_BASE_URL:-}" ]; then
    if [ "${NODE_TLS_REJECT_UNAUTHORIZED:-1}" = "0" ]; then
        ok "9. ANTHROPIC_BASE_URL set ($ANTHROPIC_BASE_URL); TLS-bypass on"
    else
        warn "9. ANTHROPIC_BASE_URL set but NODE_TLS_REJECT_UNAUTHORIZED is unset/1"
        hint "  Many corp proxies use self-signed certs. If the run dies with"
        hint "  'unable to verify the first certificate' or claude CLI exits"
        hint "  immediately, add: export NODE_TLS_REJECT_UNAUTHORIZED=0"
        hint "  Also set ANTHROPIC_AUTH_TOKEN=\$ANTHROPIC_API_KEY if the proxy"
        hint "  expects 'Authorization: Bearer ...' instead of 'x-api-key'."
    fi
fi
if [ -n "${OPENAI_BASE_URL:-}" ]; then
    if [ "${INFERENCE_OPTIMIZER_OPENAI_VERIFY_SSL:-1}" = "0" ]; then
        ok "9b. OPENAI_BASE_URL set ($OPENAI_BASE_URL); SSL-verify off"
    else
        warn "9b. OPENAI_BASE_URL set but INFERENCE_OPTIMIZER_OPENAI_VERIFY_SSL unset"
        hint "  For corp/self-signed proxies: export INFERENCE_OPTIMIZER_OPENAI_VERIFY_SSL=0"
    fi
fi

# ---------------------------------------------------------------------------
# 10. Session root writable
# ---------------------------------------------------------------------------
mkdir -p "$SESSION_ROOT" 2>/dev/null
if [ -d "$SESSION_ROOT" ] && [ -w "$SESSION_ROOT" ]; then
    AVAIL=$(df -BG "$SESSION_ROOT" 2>/dev/null | awk 'NR==2 {print $4}')
    ok "10. session root $SESSION_ROOT writable (free=$AVAIL)"
else
    fail "10. session root $SESSION_ROOT not writable"
    hint "  export INFERENCE_OPTIMIZER_SESSION_ROOT=/tmp/io-sessions"
fi

# ---------------------------------------------------------------------------
# 11. node + claude CLI (for --backend claude). --auto-install is fine
# ---------------------------------------------------------------------------
NODE_OK=no
CLAUDE_OK=no
command -v node >/dev/null 2>&1 && {
    NV=$(node --version 2>/dev/null)
    case "$NV" in
        v18.*|v19.*|v2[0-9].*|v[3-9][0-9].*) NODE_OK=yes ;;
    esac
}
command -v claude >/dev/null 2>&1 && CLAUDE_OK=yes
if [ -x "$HOME/.cache/inference-optimizer/npm-prefix/bin/claude" ]; then
    CLAUDE_OK=yes  # the bootstrap-installed copy
fi
if [ "$NODE_OK" = "yes" ] && [ "$CLAUDE_OK" = "yes" ]; then
    ok "11. node+claude CLI available"
elif [ "${INFERENCE_OPTIMIZER_AUTO_INSTALL:-0}" = "1" ]; then
    ok "11. node+claude missing; --auto-install ON (will fetch on launch)"
else
    warn "11. node+claude CLI missing"
    hint "  Pass --auto-install to the CLI, or:"
    hint "  export INFERENCE_OPTIMIZER_AUTO_INSTALL=1"
fi

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo
echo "============================================================"
echo "Pre-flight summary: ${OK_COUNT} ok, ${WARN_COUNT} warn, ${FAIL_COUNT} fail"
echo "============================================================"
if [ ${#HINTS[@]} -gt 0 ]; then
    echo "Hints:"
    for h in "${HINTS[@]}"; do
        printf "%s\n" "$h"
    done
fi
echo

if [ "$FAIL_COUNT" -gt 0 ]; then
    printf "${RED}Pre-flight FAILED — fix above issues before launching.${NC}\n"
    exit 1
fi
printf "${GREEN}Pre-flight PASSED — safe to launch.${NC}\n"
exit 0
