#!/usr/bin/env bash
# Claw Health Watchdog — monitors Primus-Claw and relaunches if unhealthy
#
# Usage:
#   nohup bash claw_watchdog.sh > /tmp/claw_watchdog.log 2>&1 &

set -euo pipefail

CLAW_DIR="${CLAW_DIR:-/shared_nfs/nehaprakriya/Primus-Claw/Claw}"
CLAW_URL="${CLAW_URL:-http://localhost:8000}"
EXECUTOR_URL="${EXECUTOR_URL:-http://localhost:8100}"
AUTH_PROXY_URL="${AUTH_PROXY_URL:-http://127.0.0.1:4002}"
AUTH_PROXY_SCRIPT="${AUTH_PROXY_SCRIPT:-/shared_nfs/nehaprakriya/Primus-Claw/OOB/auth_proxy.py}"
POLL_INTERVAL_S="${POLL_INTERVAL_S:-30}"
FAIL_THRESHOLD="${FAIL_THRESHOLD:-3}"
LOG_PREFIX="[claw-watchdog]"

export BACKEND_VENV_DIR="${BACKEND_VENV_DIR:-/tmp/primus-claw-backend-venv}"
export PROXY_AUTH_TOKEN="${PROXY_AUTH_TOKEN:-${ANTHROPIC_AUTH_TOKEN}}"
export LLM_PROXY_HOST="${LLM_PROXY_HOST:-oci-slc.example-internal-host.invalid}"
export AUTH_PROXY_PORT="${AUTH_PROXY_PORT:-4002}"

consecutive_fails=0
total_restarts=0

log() { echo "$(date '+%Y-%m-%d %H:%M:%S') $LOG_PREFIX $*"; }

check_health() {
    local url="$1"
    local resp
    resp=$(curl -s -o /dev/null -w '%{http_code}' --max-time 10 "$url/health" 2>/dev/null) || resp="000"
    [[ "$resp" == "200" ]]
}

restart_auth_proxy() {
    log "Checking auth proxy..."
    if check_health "$AUTH_PROXY_URL"; then
        return 0
    fi

    log "Auth proxy down — restarting on port $AUTH_PROXY_PORT"
    # Kill stale
    local pids
    pids=$(lsof -ti ":$AUTH_PROXY_PORT" 2>/dev/null) || pids=$(ss -tlnp "sport = :$AUTH_PROXY_PORT" 2>/dev/null | grep -oP 'pid=\K[0-9]+' | sort -u | xargs) || true
    [[ -n "$pids" ]] && kill -9 $pids 2>/dev/null || true
    sleep 1

    nohup python3 "$AUTH_PROXY_SCRIPT" >> /tmp/auth_proxy.log 2>&1 &
    local proxy_pid=$!
    echo "$proxy_pid" > /tmp/auth_proxy_pid.txt
    sleep 2

    if check_health "$AUTH_PROXY_URL"; then
        log "Auth proxy started (PID $proxy_pid)"
        return 0
    else
        log "ERROR: Auth proxy failed to start"
        return 1
    fi
}

restart_backend_direct() {
    # start_backend.sh hangs on `uv sync` over NFS. When the venv already
    # exists, skip start_all.sh entirely and launch uvicorn directly.
    log "Starting backend directly (bypassing uv sync)..."

    local backend_host="${BACKEND_HOST:-0.0.0.0}"
    local backend_port="${BACKEND_PORT:-8000}"

    # Kill stale backend on port
    local pids
    if command -v lsof &>/dev/null; then
        pids=$(lsof -ti ":$backend_port" 2>/dev/null) || true
    else
        pids=$(ss -tlnp "sport = :$backend_port" 2>/dev/null | grep -oP 'pid=\K[0-9]+' | sort -u | xargs) || true
    fi
    [[ -n "$pids" ]] && kill -9 $pids 2>/dev/null || true
    sleep 1

    local venv_uvicorn="/tmp/claw-backend-venv/bin/uvicorn"
    if [[ ! -x "$venv_uvicorn" ]]; then
        log "Backend venv missing at $venv_uvicorn — rebuilding with uv sync"
        (cd "$CLAW_DIR/backend" && UV_PROJECT_ENVIRONMENT=/tmp/claw-backend-venv uv sync 2>&1) || true
        if [[ ! -x "$venv_uvicorn" ]]; then
            log "ERROR: $venv_uvicorn still not found after rebuild"
            return 1
        fi
    fi

    (
        cd "$CLAW_DIR/backend"
        set -a
        [[ -f "$CLAW_DIR/.env" ]] && source "$CLAW_DIR/.env"
        set +a
        export PYTHONPATH="$CLAW_DIR:${PYTHONPATH:-}"
        nohup "$venv_uvicorn" app.main:app --host "$backend_host" --port "$backend_port" \
            >> "$CLAW_DIR/logs/backend.log" 2>&1 &
        echo "$!" > "$CLAW_DIR/.backend.pid"
    )
    sleep 3

    if check_health "$CLAW_URL"; then
        log "Backend started directly (PID $(cat "$CLAW_DIR/.backend.pid" 2>/dev/null))"
        return 0
    fi
    log "ERROR: Backend failed to start directly"
    return 1
}

restart_executor_direct() {
    log "Starting executor directly (bypassing uv sync)..."

    local executor_port="${EXECUTOR_PORT:-8100}"

    # Kill stale executor on port
    local pids
    if command -v lsof &>/dev/null; then
        pids=$(lsof -ti ":$executor_port" 2>/dev/null) || true
    else
        pids=$(ss -tlnp "sport = :$executor_port" 2>/dev/null | grep -oP 'pid=\K[0-9]+' | sort -u | xargs) || true
    fi
    [[ -n "$pids" ]] && kill -9 $pids 2>/dev/null || true
    sleep 1

    local executor_dir="$CLAW_DIR/executor"
    local venv_uvicorn="$executor_dir/.venv/bin/uvicorn"
    if [[ ! -x "$venv_uvicorn" ]]; then
        log "ERROR: $venv_uvicorn not found — cannot start executor directly"
        return 1
    fi

    (
        cd "$executor_dir"
        set -a
        [[ -f "$CLAW_DIR/.env" ]] && source "$CLAW_DIR/.env"
        set +a
        nohup "$venv_uvicorn" app.main:app --host 0.0.0.0 --port "$executor_port" \
            >> "$CLAW_DIR/logs/executor.log" 2>&1 &
        echo "$!" > "$CLAW_DIR/.executor.pid"
    )
    sleep 3

    if check_health "$EXECUTOR_URL"; then
        log "Executor started directly (PID $(cat "$CLAW_DIR/.executor.pid" 2>/dev/null))"
        return 0
    fi
    log "ERROR: Executor failed to start directly"
    return 1
}

restart_claw() {
    log "RESTARTING Claw (restart #$((total_restarts + 1)))..."

    # Restart auth proxy first
    restart_auth_proxy

    # Kill any existing backend/executor
    local backend_pid executor_pid
    backend_pid=$(cat "$CLAW_DIR/.backend.pid" 2>/dev/null) || true
    executor_pid=$(cat "$CLAW_DIR/.executor.pid" 2>/dev/null) || true

    [[ -n "$backend_pid" ]] && kill "$backend_pid" 2>/dev/null || true
    [[ -n "$executor_pid" ]] && kill "$executor_pid" 2>/dev/null || true
    sleep 2

    # Kill anything on ports 8000/8100
    for port in 8000 8100; do
        local pids
        if command -v lsof &>/dev/null; then
            pids=$(lsof -ti ":$port" 2>/dev/null) || true
        else
            pids=$(ss -tlnp "sport = :$port" 2>/dev/null | grep -oP 'pid=\K[0-9]+' | sort -u | xargs) || true
        fi
        [[ -n "$pids" ]] && kill -9 $pids 2>/dev/null || true
    done
    sleep 2

    # Direct restart: launch backend and executor individually, bypassing
    # start_all.sh which runs `uv sync` and hangs on NFS.
    local backend_up=false executor_up=false

    if ! check_health "$CLAW_URL"; then
        restart_backend_direct && backend_up=true
    else
        backend_up=true
    fi

    if ! check_health "$EXECUTOR_URL"; then
        restart_executor_direct && executor_up=true
    else
        executor_up=true
    fi

    # If direct restart failed, fall back to start_all.sh with a timeout
    if ! $backend_up || ! $executor_up; then
        log "Direct restart incomplete (backend=$backend_up executor=$executor_up) — trying start_all.sh with timeout"
        cd "$CLAW_DIR"
        timeout 90 bash start_all.sh --with-executor --no-watch 2>&1 | while IFS= read -r line; do
            log "  startup: $line"
        done
        local rc=${PIPESTATUS[0]}
        if [[ "$rc" -ne 0 ]]; then
            log "ERROR: start_all.sh exited $rc (or timed out)"
        fi
    fi

    sleep 3

    if check_health "$CLAW_URL" && check_health "$EXECUTOR_URL"; then
        total_restarts=$((total_restarts + 1))
        log "Claw restarted successfully (total restarts: $total_restarts)"
        return 0
    else
        log "ERROR: Claw still unhealthy after restart"
        return 1
    fi
}

# Startup check
log "Starting Claw watchdog (poll=${POLL_INTERVAL_S}s, threshold=${FAIL_THRESHOLD})"
log "  CLAW_DIR=$CLAW_DIR"
log "  CLAW_URL=$CLAW_URL"

restart_auth_proxy || log "WARNING: Auth proxy launch failed"

if ! check_health "$CLAW_URL"; then
    log "Claw not healthy at startup — launching now"
    restart_claw || log "WARNING: Initial launch failed, will keep retrying"
fi

INFERENCE_SERVER_PORT="${INFERENCE_SERVER_PORT:-8888}"
BENCH_MAX_AGE_S="${BENCH_MAX_AGE_S:-660}"
INFERENCE_SERVER_MODEL="${INFERENCE_SERVER_MODEL:-/shared_nfs/models/gpt-oss-120b}"
GPU_MIN_FREE_PCT="${GPU_MIN_FREE_PCT:-90}"

kill_orphaned_claw_agents() {
    # Claw agents from previous marathon sessions can linger and interfere
    # (starting servers, running benchmarks, modifying files). Kill any
    # claude processes older than MAX_AGENT_AGE_S that aren't the executor.
    local MAX_AGENT_AGE_S="${MAX_AGENT_AGE_S:-3600}"
    local executor_pid
    executor_pid=$(pgrep -f 'uvicorn app.main:app.*8100' 2>/dev/null | head -1) || true

    for pid in $(pgrep -f 'claude' 2>/dev/null); do
        # Skip the executor process itself
        [[ "$pid" == "$executor_pid" ]] && continue
        # Skip non-claude processes (bash children)
        comm=$(cat "/proc/$pid/comm" 2>/dev/null) || continue
        [[ "$comm" != "claude" ]] && continue

        local age_s now
        now=$(date +%s)
        age_s=$(( now - $(stat -c %Y "/proc/$pid" 2>/dev/null || echo "$now") ))
        if [[ "$age_s" -ge "$MAX_AGENT_AGE_S" ]]; then
            log "Killing orphaned claude agent PID $pid (age ${age_s}s > ${MAX_AGENT_AGE_S}s)"
            # Kill the process group to catch child shells/find commands
            kill -9 -"$pid" 2>/dev/null || kill -9 "$pid" 2>/dev/null || true
        fi
    done

    # Also kill stuck 'find /' commands from Claw agents exploring the filesystem
    for pid in $(pgrep -f '^find /' 2>/dev/null); do
        local age_s now
        now=$(date +%s)
        age_s=$(( now - $(stat -c %Y "/proc/$pid" 2>/dev/null || echo "$now") ))
        if [[ "$age_s" -ge 300 ]]; then
            log "Killing stuck find process PID $pid (age ${age_s}s)"
            kill -9 "$pid" 2>/dev/null || true
        fi
    done
}

kill_stale_benchmarks() {
    local pids age_s now
    now=$(date +%s)
    for pid in $(pgrep -f 'benchmark_serving' 2>/dev/null); do
        age_s=$(( now - $(stat -c %Y "/proc/$pid" 2>/dev/null || echo "$now") ))
        if [[ "$age_s" -ge "$BENCH_MAX_AGE_S" ]]; then
            log "Killing stale benchmark PID $pid (age ${age_s}s > ${BENCH_MAX_AGE_S}s)"
            kill -9 "$pid" 2>/dev/null || true
        fi
    done
}

check_inference_server() {
    local resp
    resp=$(curl -s -o /dev/null -w '%{http_code}' --max-time 5 "http://localhost:${INFERENCE_SERVER_PORT}/health" 2>/dev/null) || resp="000"
    [[ "$resp" == "200" ]]
}

deep_check_inference_server() {
    # Verify the server can actually produce tokens, not just respond to /health.
    if ! check_inference_server; then
        return 1
    fi
    local body
    body=$(curl -sf --max-time 30 \
        -H 'Content-Type: application/json' \
        -d "{\"model\":\"$INFERENCE_SERVER_MODEL\",\"prompt\":\"test\",\"max_tokens\":1}" \
        "http://localhost:${INFERENCE_SERVER_PORT}/v1/completions" 2>/dev/null) || return 1
    echo "$body" | grep -q '"choices"' 2>/dev/null
}

detect_gpu_zombies() {
    local kfd_dir="/sys/class/kfd/kfd/proc"
    [[ -d "$kfd_dir" ]] || return 0
    local has_zombies=false
    for d in "$kfd_dir"/*/; do
        local pid
        pid=$(basename "$d")
        [[ "$pid" =~ ^[0-9]+$ ]] || continue
        if [[ ! -d "/proc/$pid" ]]; then
            log "WARNING: Zombie GPU allocation — KFD PID $pid is dead but holding VRAM"
            has_zombies=true
        fi
    done
    $has_zombies && return 1 || return 0
}

find_clean_gpu() {
    # Print the ID of the first GPU with >= GPU_MIN_FREE_PCT% free VRAM.
    local output gpu_id used total free_pct
    output=$(rocm-smi --showmeminfo vram 2>/dev/null) || return 1

    local -a ids=() totals=() useds=()
    while IFS= read -r line; do
        if [[ "$line" =~ GPU\[([0-9]+)\].*VRAM\ Total\ Memory.*:\ ([0-9]+) ]]; then
            ids+=("${BASH_REMATCH[1]}")
            totals+=("${BASH_REMATCH[2]}")
        elif [[ "$line" =~ GPU\[([0-9]+)\].*VRAM\ Total\ Used.*:\ ([0-9]+) ]]; then
            useds+=("${BASH_REMATCH[2]}")
        fi
    done <<< "$output"

    for i in "${!ids[@]}"; do
        used=${useds[$i]:-0}
        total=${totals[$i]:-1}
        if (( total > 0 )); then
            free_pct=$(awk "BEGIN {printf \"%.0f\", (1.0 - $used / $total) * 100}")
            if (( free_pct >= GPU_MIN_FREE_PCT )); then
                echo "${ids[$i]}"
                return 0
            fi
        fi
    done
    return 1
}

# Main loop
while true; do
    sleep "$POLL_INTERVAL_S"

    backend_ok=true
    executor_ok=true
    proxy_ok=true

    if ! check_health "$CLAW_URL"; then
        backend_ok=false
    fi
    if ! check_health "$EXECUTOR_URL"; then
        executor_ok=false
    fi
    if ! check_health "$AUTH_PROXY_URL"; then
        proxy_ok=false
        restart_auth_proxy
    fi

    kill_stale_benchmarks
    kill_orphaned_claw_agents

    if ! detect_gpu_zombies; then
        clean_gpu=$(find_clean_gpu 2>/dev/null) || clean_gpu=""
        if [[ -n "$clean_gpu" ]]; then
            log "GPU zombies present but GPU $clean_gpu is clean — serve_tp1.sh should auto-select it"
        else
            log "ERROR: GPU zombies present and NO clean GPU available. Pod restart may be required."
        fi
    fi

    if ! check_inference_server; then
        log "WARNING: Inference server on port $INFERENCE_SERVER_PORT not responding to /health"
    elif ! deep_check_inference_server; then
        log "WARNING: Inference server responds to /health but FAILS inference probe — server may be corrupted"
    fi

    # Detect rogue vllm servers started by Claw agents (not by the marathon).
    # The marathon writes .server_owner_pid when it starts the server.
    local owner_pid_file
    for d in /shared_nfs/nehaprakriya/Agentic-InferenceX/*/sessions/*/benchmarks/.server_owner_pid; do
        [[ -f "$d" ]] || continue
        owner_pid_file="$d"
        break
    done
    if [[ -n "${owner_pid_file:-}" ]]; then
        local owner_pid
        owner_pid=$(cat "$owner_pid_file" 2>/dev/null) || true
        for vllm_pid in $(pgrep -f 'vllm serve' 2>/dev/null); do
            local ppid
            ppid=$(ps -o ppid= -p "$vllm_pid" 2>/dev/null | tr -d ' ') || continue
            # If the vllm was started by a claude agent (not by marathon/workload),
            # it's a rogue. Check if the parent chain includes claude.
            if ps -o comm= -p "$ppid" 2>/dev/null | grep -q 'claude\|bash'; then
                local parent_cmd
                parent_cmd=$(cat "/proc/$ppid/cmdline" 2>/dev/null | tr '\0' ' ' | head -c 200) || true
                if echo "$parent_cmd" | grep -q 'claude\|shell-snapshot'; then
                    log "WARNING: Rogue vllm server PID $vllm_pid started by Claw agent (parent $ppid) — killing"
                    kill -9 "$vllm_pid" 2>/dev/null || true
                fi
            fi
        done
    fi

    if $backend_ok && $executor_ok; then
        if [[ "$consecutive_fails" -gt 0 ]]; then
            log "Claw recovered (was failing for $consecutive_fails checks)"
        fi
        consecutive_fails=0
        continue
    fi

    consecutive_fails=$((consecutive_fails + 1))
    log "Health check FAILED ($consecutive_fails/$FAIL_THRESHOLD) backend=$backend_ok executor=$executor_ok proxy=$proxy_ok"

    if [[ "$consecutive_fails" -ge "$FAIL_THRESHOLD" ]]; then
        log "Threshold reached — triggering restart"
        if restart_claw; then
            consecutive_fails=0
        else
            log "Restart failed — will retry on next cycle"
            consecutive_fails=0
            sleep 30
        fi
    fi
done
