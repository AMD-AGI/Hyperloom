#!/usr/bin/env bash
# Copyright Advanced Micro Devices, Inc. All rights reserved.

# IR-3 — Cortex KB + PR Monitor reachability probe.
#
# Soft-degrade: exit 1 only signals "at least one branch unreachable".
# The cli decides what to do (auto-enable --degraded-* and continue;
# never sys.exit). Op-out by setting SKIP_KB_PROBE / SKIP_PR_PROBE=1
# for the corresponding branch (the cli sets these when the operator
# passed --degraded-kb / --degraded-pr).
#
# Writes a marker JSON to $USER_DATA_PATH/runtime/cortex/.kb_preflight.json
# capturing reachability + skip + failure_reason per branch.

set -u

: "${CORTEX_KB_URL:=}"
: "${PR_MONITOR_URL:=http://primus-cortex-pr-api.primus-cortex.svc.cluster.local/v1}"
# Capture whether USER_DATA_PATH was provided BEFORE applying the default so we
# can warn loudly on the silent fallback. ${VAR:+1} is empty when VAR is unset
# or empty, which is exactly the case the := default below would absorb.
_user_data_was_set="${USER_DATA_PATH:+1}"
: "${USER_DATA_PATH:=/workspace/hyperloom}"
if [ -z "${_user_data_was_set}" ]; then
  echo "[install WARN] USER_DATA_PATH not set; defaulting to /workspace/hyperloom. Set USER_DATA_PATH to persist artifacts under your data root." >&2
fi
: "${KB_SERVICE_TOKEN:=}"
: "${SKIP_KB_PROBE:=}"
: "${SKIP_PR_PROBE:=}"

marker_dir="${USER_DATA_PATH}/runtime/cortex"
mkdir -p "${marker_dir}"
marker="${marker_dir}/.kb_preflight.json"

kb_reachable="false"
kb_skipped="false"
kb_failure_reason=""

pr_reachable="false"
pr_skipped="false"
pr_failure_reason=""

# Retry helper: probe_curl URL timeout-sec attempts label
probe_curl() {
    local url="$1"; shift
    local timeout="$1"; shift
    local attempts="$1"; shift
    local label="$1"; shift
    local sleeps=(1 3 5)

    local code
    local body
    local i=0
    local last_failure=""
    while [ "${i}" -lt "${attempts}" ]; do
        body="$(curl --silent --max-time "${timeout}" -w '\n__HTTP_CODE__:%{http_code}' "${url}" 2>/dev/null || true)"
        code="$(printf '%s' "${body}" | sed -n 's/^__HTTP_CODE__://p' | tail -n1)"
        body="$(printf '%s' "${body}" | sed -e 's/__HTTP_CODE__:[0-9]*$//')"
        if [ "${code:-000}" = "200" ]; then
            if [ "${label}" = "kb" ]; then
                if printf '%s' "${body}" | grep -q '"status"[[:space:]]*:[[:space:]]*"ok"'; then
                    kb_reachable="true"
                    return 0
                fi
                # 200 but no status:ok payload → treat as unreachable
                last_failure="200_without_status_ok"
            else
                pr_reachable="true"
                return 0
            fi
        elif [ "${code:-000}" = "401" ] && [ "${label}" = "kb" ]; then
            if [ -n "${KB_SERVICE_TOKEN}" ]; then
                # Token configured + server requires auth → treat as
                # reachable (H2 path). H1 is anonymous so a 401 with
                # no token configured is a real failure.
                kb_reachable="true"
                return 0
            fi
            last_failure="missing_token"
        else
            last_failure="${code:-timeout}"
        fi
        i=$((i + 1))
        if [ "${i}" -lt "${attempts}" ]; then
            sleep "${sleeps[$((i - 1))]:-5}"
        fi
    done
    if [ "${label}" = "kb" ]; then
        kb_failure_reason="${last_failure:-unknown}"
    else
        pr_failure_reason="${last_failure:-unknown}"
    fi
    return 1
}

# No URL configured (operator didn't pass --cortex-kb-url / export
# CORTEX_KB_URL) → there is no remote KB to probe. Mark the branch
# skipped so the cli stays local-only instead of soft-degrading on an
# unreachable hard-coded default (which no longer exists).
if [ -n "${SKIP_KB_PROBE}" ] || [ -z "${CORTEX_KB_URL}" ]; then
    kb_skipped="true"
else
    probe_curl "${CORTEX_KB_URL%/}/health" 15 3 "kb" || true
fi

if [ -n "${SKIP_PR_PROBE}" ]; then
    pr_skipped="true"
else
    probe_curl "${PR_MONITOR_URL%/}/healthz" 5 3 "pr" || true
fi

# Emit marker (pure shell — no jq dependency).
{
    printf '{'
    printf '"kb_reachable":%s,' "${kb_reachable}"
    printf '"pr_reachable":%s,' "${pr_reachable}"
    printf '"kb_skipped":%s,' "${kb_skipped}"
    printf '"pr_skipped":%s' "${pr_skipped}"
    if [ -n "${kb_failure_reason}" ]; then
        printf ',"kb_failure_reason":"%s"' "${kb_failure_reason}"
    fi
    if [ -n "${pr_failure_reason}" ]; then
        printf ',"pr_failure_reason":"%s"' "${pr_failure_reason}"
    fi
    printf '}\n'
} > "${marker}"

# Exit 1 only if at least one NOT-skipped branch is unreachable.
if [ "${kb_skipped}" = "false" ] && [ "${kb_reachable}" = "false" ]; then
    exit 1
fi
if [ "${pr_skipped}" = "false" ] && [ "${pr_reachable}" = "false" ]; then
    exit 1
fi
exit 0
