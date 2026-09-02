#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# IR-3 — KB Store-hosted PR Monitor reachability probe.
#
# Soft-degrade: exit 1 only signals "PR Monitor branch unreachable".
# The cli decides what to do (auto-enable --degraded-pr and continue;
# never sys.exit). Opt-out by setting SKIP_PR_PROBE=1 (the cli sets this
# when the operator passed --degraded-pr).
#
# The probe is derived directly from
# ${KB_STORE_URL}/pr-monitor/v1/healthz.
#
# Recipe KB (local + gbrain) has no remote reachability probe; the cli
# records recipe-KB enablement from --degraded-kb directly.
#
# Writes a marker JSON to $USER_DATA_PATH/runtime/recipe_kb/.kb_preflight.json
# capturing reachability + skip + failure_reason per branch.

set -u

# Legacy marker fields: kb_* are always skipped (no remote recipe-KB probe).
kb_reachable="false"
kb_skipped="true"
kb_failure_reason=""

: "${KB_STORE_URL:=}"
_user_data_was_set="${USER_DATA_PATH:+1}"
# Container images ship a writable /workspace; a bare-metal host off root has
# neither it nor permission to create it, so the mkdir below would abort.
_default_workspace_root() {
  # The nearest existing ancestor decides: -w is false for a path that does not
  # exist yet, which would divert root off a /workspace it can still create.
  _ws_probe=/workspace
  while [ ! -e "$_ws_probe" ] && [ "$_ws_probe" != / ]; do _ws_probe=$(dirname "$_ws_probe"); done
  if [ -w "$_ws_probe" ]; then printf '%s' /workspace/hyperloom; else printf '%s' "$(pwd -P)/session"; fi
}
: "${USER_DATA_PATH:=$(_default_workspace_root)}"
if [ -z "${_user_data_was_set}" ]; then
  echo "[install WARN] USER_DATA_PATH not set; defaulting to ${USER_DATA_PATH}. Set USER_DATA_PATH to persist artifacts under your data root." >&2
fi
: "${SKIP_PR_PROBE:=}"

marker_dir="${USER_DATA_PATH}/runtime/recipe_kb"
mkdir -p "${marker_dir}"
marker="${marker_dir}/.kb_preflight.json"

pr_reachable="false"
pr_skipped="false"
pr_failure_reason=""

probe_curl() {
    local url="$1"; shift
    local timeout="$1"; shift
    local attempts="$1"; shift
    local sleeps=(1 3 5)

    local code
    local body
    local i=0
    local last_failure=""
    while [ "${i}" -lt "${attempts}" ]; do
        body="$(curl --silent --max-time "${timeout}" -w '\n__HTTP_CODE__:%{http_code}' "${url}" 2>/dev/null || true)"
        code="$(printf '%s' "${body}" | sed -n 's/^__HTTP_CODE__://p' | tail -n1)"
        if [ "${code:-000}" = "200" ]; then
            pr_reachable="true"
            return 0
        else
            last_failure="${code:-timeout}"
        fi
        i=$((i + 1))
        if [ "${i}" -lt "${attempts}" ]; then
            sleep "${sleeps[$((i - 1))]:-5}"
        fi
    done
    pr_failure_reason="${last_failure:-unknown}"
    return 1
}

if [ -n "${SKIP_PR_PROBE}" ] || [ -z "${KB_STORE_URL}" ]; then
    pr_skipped="true"
else
    probe_curl "${KB_STORE_URL%/}/pr-monitor/v1/healthz" 5 3 || true
fi

{
    printf '{'
    printf '"kb_reachable":%s,' "${kb_reachable}"
    printf '"pr_reachable":%s,' "${pr_reachable}"
    printf '"kb_skipped":%s,' "${kb_skipped}"
    printf '"pr_skipped":%s' "${pr_skipped}"
    if [ -n "${pr_failure_reason}" ]; then
        printf ',"pr_failure_reason":"%s"' "${pr_failure_reason}"
    fi
    printf '}\n'
} > "${marker}"

if [ "${pr_skipped}" = "false" ] && [ "${pr_reachable}" = "false" ]; then
    exit 1
fi
exit 0
