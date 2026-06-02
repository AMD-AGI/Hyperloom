#!/usr/bin/env bash
# Launch a fresh vllm/sglang server on the RayJob head pod under nohup.
#
# Submitted via Ray Dashboard REST by ``multi_node restart-server`` as
# the second phase of a server swap. Designed to EXIT QUICKLY (after
# the launch + optional health check) so the Ray Dashboard job's status
# becomes SUCCEEDED — the actual server keeps running because we
# nohup'd it and orphaned the parent.
#
# Usage:
#   launch_server.sh FRAMEWORK MODEL TP PID_FILE LOG_FILE [--wait-health|--no-wait-health] [-- EXTRA_ARGS...]
#     FRAMEWORK   sglang | vllm
#     MODEL       model path or HF id
#     TP          tensor-parallel size (int)
#     PID_FILE    where to write the new server pid (consumed by kill_server.sh next time)
#     LOG_FILE    where to redirect server stdout+stderr
#     --wait-health     poll http://127.0.0.1:8888/health up to 60s before returning success (default)
#     --no-wait-health  exit immediately after launching (use when you'll poll /health from outside)
#     EXTRA_ARGS  forwarded verbatim to the framework launcher
#
# IMPORTANT: the inference server port is FIXED at 8888 — this matches
# the SaFE Service's targetPort that brain wires up. Do NOT make 8888
# configurable here; both ends would need to change in lock-step.

set -euo pipefail

if [ "$#" -lt 5 ]; then
    echo "launch_server.sh: usage: $0 FRAMEWORK MODEL TP PID_FILE LOG_FILE [--wait-health|--no-wait-health] [-- EXTRA_ARGS...]" >&2
    exit 2
fi

FRAMEWORK="$1"; shift
MODEL="$1"; shift
TP="$1"; shift
PID_FILE="$1"; shift
LOG_FILE="$1"; shift

WAIT_HEALTH=1
EXTRA=()
while [ "$#" -gt 0 ]; do
    case "$1" in
        --wait-health)    WAIT_HEALTH=1; shift ;;
        --no-wait-health) WAIT_HEALTH=0; shift ;;
        --) shift; EXTRA=("$@"); break ;;
        *) echo "launch_server.sh: unknown arg before --: $1" >&2; exit 2 ;;
    esac
done

PORT=8888

# Source the bootstrap-rendered env so PATH points at /opt/venv and
# *_API_KEY / *_BASE_URL are visible to the launched server (some
# frameworks read these for their own LLM clients / telemetry).
if [ -f /etc/profile.d/hyperloom-env.sh ]; then
    # shellcheck disable=SC1091
    source /etc/profile.d/hyperloom-env.sh
fi

mkdir -p "$(dirname "$LOG_FILE")" "$(dirname "$PID_FILE")"

case "$FRAMEWORK" in
    sglang)
        # Backend-agnostic. EXTRA contains things like
        # `--mem-fraction-static 0.85 --enable-torch-compile ...`.
        CMD=(python3 -m sglang.launch_server
             --model-path "$MODEL"
             --tp "$TP"
             --port "$PORT"
             --host 0.0.0.0)
        ;;
    vllm)
        CMD=(vllm serve "$MODEL"
             --tensor-parallel-size "$TP"
             --port "$PORT"
             --host 0.0.0.0)
        ;;
    *)
        echo "launch_server.sh: unsupported framework '$FRAMEWORK' (use sglang or vllm)" >&2
        exit 2
        ;;
esac

# Append any extra args verbatim.
if [ "${#EXTRA[@]}" -gt 0 ]; then
    CMD+=("${EXTRA[@]}")
fi

echo "launch_server.sh: framework=$FRAMEWORK model=$MODEL tp=$TP port=$PORT"
echo "launch_server.sh: log=$LOG_FILE pid=$PID_FILE"
echo "launch_server.sh: cmd=${CMD[*]}"

# nohup + setsid so the server survives after this script exits.
# `disown` belt-and-suspenders for some shells.
nohup setsid "${CMD[@]}" > "$LOG_FILE" 2>&1 &
NEW_PID=$!
echo "$NEW_PID" > "$PID_FILE"
disown "$NEW_PID" 2>/dev/null || true
echo "launch_server.sh: spawned pid=$NEW_PID"

if [ "$WAIT_HEALTH" -eq 1 ]; then
    # 12 probes * 5s = up to 60s. Cold start of large MoE can take much
    # longer; the agent should call again with --no-wait-health for the
    # cold case and probe /health from the sandbox via the ClusterIP.
    for i in $(seq 1 12); do
        sleep 5
        if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
            echo "launch_server.sh: health OK after ${i} probes"
            exit 0
        fi
        # Cheap liveness check: if the process died early we should fail
        # fast instead of waiting the full 60s.
        if ! kill -0 "$NEW_PID" 2>/dev/null; then
            echo "launch_server.sh: server pid=$NEW_PID exited early; check $LOG_FILE" >&2
            exit 1
        fi
    done
    echo "launch_server.sh: WARN — /health did not pass within 60s; server still starting? check $LOG_FILE" >&2
    # Don't fail here — the server may be in the middle of weight loading.
    # The agent can poll /health from the sandbox separately.
fi
echo "launch_server.sh: OK (server detached, pid=$NEW_PID)"
