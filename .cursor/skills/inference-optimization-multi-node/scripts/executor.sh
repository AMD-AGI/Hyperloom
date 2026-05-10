#!/usr/bin/env bash
# =============================================================================
# executor.sh — RayJob REST execution helpers
#
# Dispatches GPU-side commands to a running RayJob through Ray Dashboard REST.
# This intentionally avoids Ray Client (`ray://<head>:10001`) from the sandbox.
#
# Required env:
#   MODE              — "remote"
#   HEAD_IP           — RayJob head pod IP, OR
#   RAY_DASHBOARD_URL — full dashboard URL, e.g. http://<head-ip>:8265
#
# Usage:
#   source executor.sh
#   exec_on_gpu "bash \$SCRIPTS_DIR/run_baseline.sh" 1800
#   sid=$(exec_on_gpu_bg "bash \$SCRIPTS_DIR/run_profile.sh")
#   exec_on_gpu_status "$sid"
#   exec_on_gpu_logs "$sid"
# =============================================================================

SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

_ray_dashboard_url() {
    if [ -n "${RAY_DASHBOARD_URL:-}" ]; then
        printf '%s' "${RAY_DASHBOARD_URL%/}"
    elif [ -n "${HEAD_IP:-}" ]; then
        printf 'http://%s:8265' "$HEAD_IP"
    elif [ -n "${RAY_HEAD_IP:-}" ]; then
        printf 'http://%s:8265' "$RAY_HEAD_IP"
    else
        echo "ERROR: set HEAD_IP, RAY_HEAD_IP, or RAY_DASHBOARD_URL for remote execution" >&2
        return 1
    fi
}

_json_escape() {
    if command -v node >/dev/null 2>&1; then
        NODE_VALUE="$1" node -e 'process.stdout.write(JSON.stringify(process.env.NODE_VALUE || ""))'
    else
        # Minimal fallback. Prefer node because commands can contain quotes/newlines.
        printf '"%s"' "$(printf '%s' "$1" | sed 's/\\/\\\\/g; s/"/\\"/g')"
    fi
}

_json_field() {
    local field="$1"
    if command -v node >/dev/null 2>&1; then
        FIELD="$field" node -e '
const fs = require("fs");
const input = fs.readFileSync(0, "utf8");
try {
  const obj = JSON.parse(input);
  const value = obj[process.env.FIELD] ?? "";
  process.stdout.write(String(value));
} catch {
  process.exit(1);
}'
    else
        sed -n "s/.*\"$field\"[[:space:]]*:[[:space:]]*\"\([^\"]*\)\".*/\1/p" | head -1
    fi
}

exec_on_gpu_submit() {
    local cmd="$1"
    local url
    url="$(_ray_dashboard_url)" || return 1

    local escaped
    escaped="$(_json_escape "$cmd")" || return 1

    local response
    response=$(curl -sS -X POST "${url}/api/jobs/" \
        -H "Content-Type: application/json" \
        -d "{\"entrypoint\": ${escaped}}") || return 1

    local sid
    sid=$(printf '%s' "$response" | _json_field submission_id)
    if [ -z "$sid" ] || [ "$sid" = "null" ]; then
        echo "ERROR: Ray Dashboard did not return submission_id" >&2
        echo "$response" >&2
        return 1
    fi
    printf '%s\n' "$sid"
}

exec_on_gpu_status() {
    local submission_id="$1"
    local url
    url="$(_ray_dashboard_url)" || return 1
    curl -sS "${url}/api/jobs/${submission_id}"
}

exec_on_gpu_logs() {
    local submission_id="$1"
    local url
    url="$(_ray_dashboard_url)" || return 1
    local response
    response=$(curl -sS "${url}/api/jobs/${submission_id}/logs") || return 1
    printf '%s' "$response" | _json_field logs
}

exec_on_gpu() {
    local cmd="$1"
    local timeout="${2:-3600}"

    if [ "$MODE" = "remote" ]; then
        local sid start now status_json status
        sid="$(exec_on_gpu_submit "$cmd")" || return 1
        echo "Submitted Ray job: $sid" >&2
        start=$(date +%s)

        while true; do
            status_json="$(exec_on_gpu_status "$sid")" || return 1
            status="$(printf '%s' "$status_json" | _json_field status)"
            case "$status" in
                SUCCEEDED)
                    exec_on_gpu_logs "$sid"
                    return 0
                    ;;
                FAILED|STOPPED)
                    exec_on_gpu_logs "$sid" >&2 || true
                    echo "ERROR: Ray job $sid ended with status $status" >&2
                    return 1
                    ;;
                *)
                    now=$(date +%s)
                    if [ $((now - start)) -ge "$timeout" ]; then
                        echo "ERROR: Ray job $sid timed out after ${timeout}s (last status: ${status:-unknown})" >&2
                        return 124
                    fi
                    sleep "${RAY_JOB_POLL_INTERVAL_S:-5}"
                    ;;
            esac
        done
    else
        echo "ERROR: Unknown MODE='$MODE'. Expected 'remote'." >&2
        return 1
    fi
}

exec_on_gpu_bg() {
    local cmd="$1"

    if [ "$MODE" = "remote" ]; then
        exec_on_gpu_submit "$cmd"
    else
        echo "ERROR: Unknown MODE='$MODE'. Expected 'remote'." >&2
        return 1
    fi
}
