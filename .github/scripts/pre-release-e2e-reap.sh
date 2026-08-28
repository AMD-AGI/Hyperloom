#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# Pre-release E2E: PRE-EMPT stale SaFE workloads at the very START of a run, BEFORE any
# job that touches the single self-hosted GPU runner.
#
# Why a standalone script + its own job (not the reap inside the dispatch script):
#   The dispatch reap runs in the `dispatch` job, which `needs: [resolve, build]` and
#   `runs-on: hyperloom-pre-e2e-baremetal` -- the SINGLE self-hosted runner. When a newer
#   commit supersedes an in-flight run, GitHub's concurrency.cancel-in-progress cancels
#   the old JOB but does NOT reliably stop the SaFE PyTorchJob pods it created, and the
#   old run may still occupy that one runner (and its 8 GPUs). So the new run's
#   resolve/build queue BEHIND the old run and the dispatch reap never gets to run -- a
#   deadlock where the cleanup is queued behind the very thing it must clean up.
#
#   This script runs in a `preempt` job on a GITHUB-HOSTED runner (ubuntu-latest), which
#   does NOT queue behind the busy baremetal runner. It only needs network reach to the
#   SaFE API. It fires first and every other job `needs: preempt`, so the stale pods are
#   stopped -> GPUs freed -> the old run's poll sees phase=Stopped and its legs FAIL ->
#   the old GitHub job ends as a CONSEQUENCE of the pod stopping (the correct causal
#   order), and this run's resolve/build/dispatch can then get the runner + GPUs.
#
# Pre-emption semantics: this runs BEFORE this run dispatches ANY workload, so every
# non-terminal `e2e-*` workload in this workspace is necessarily from an OLDER run and is
# safe to stop wholesale -- no VERSION_TAG self-exclusion needed (there is nothing of
# ours to exclude yet).
#
# Inputs (env):
#   SAFE_API_BASE      SaFE API base url                          (required)
#   SAFE_API_KEY       bearer token (ADMIN, to stop privileged pods) (required)
#   SAFE_WORKSPACE_ID  workspace to scope the reap to             (required)
#   SAFE_CACERT / SAFE_INSECURE  TLS to the API (CA bundle / skip-verify)
set -euo pipefail

: "${SAFE_API_BASE:?SAFE_API_BASE is required}"
: "${SAFE_API_KEY:?SAFE_API_KEY is required}"
: "${SAFE_WORKSPACE_ID:?SAFE_WORKSPACE_ID is required}"

API="${SAFE_API_BASE%/}/api/v1/workloads"
auth=(-H "Authorization: Bearer ${SAFE_API_KEY}")
tls=()
if [ -n "${SAFE_CACERT:-}" ]; then
  tls=(--cacert "$SAFE_CACERT")
elif [ "${SAFE_INSECURE:-0}" = "1" ]; then
  tls=(-k)
fi

summary() { echo "$*" | tee -a "${GITHUB_STEP_SUMMARY:-/dev/null}"; }

# List every e2e-* workload in this workspace that is NOT already terminal, and POST
# /stop to each. Resilient: a missing/unreachable API is a skip, never a hard failure
# (we must not block the run just because the reclaim couldn't reach SaFE).
reap_all_stale() {
  local resp
  resp="$(curl -sS "${tls[@]}" --max-time 30 "$API" "${auth[@]}" 2>/dev/null || true)"
  [ -n "$resp" ] || { summary "• [preempt] could not list workloads; skipping reclaim"; return 0; }
  local stale
  stale="$(printf '%s' "$resp" | jq -r --arg ws "$SAFE_WORKSPACE_ID" '
      (.items // .workloads // .)[]?
      | select(((.displayName // .name // "") | startswith("e2e-")))
      | select((.workspaceId // $ws) == $ws)
      | select((.phase // .status // "") as $p
               | (["Stopped","Failed","Succeeded","Completed","Deleted"] | index($p)) | not)
      | (.workloadId // .id)' 2>/dev/null || true)"
  [ -n "$stale" ] || { summary "• [preempt] no stale e2e workloads to reclaim"; return 0; }
  local wid code n=0
  while IFS= read -r wid; do
    [ -n "$wid" ] || continue
    code="$(curl -sS "${tls[@]}" --max-time 20 -o /dev/null -w '%{http_code}' \
      -X POST "$API/$wid/stop" "${auth[@]}" 2>/dev/null || echo 000)"
    summary "• [preempt] stopped stale workload \`$wid\` (stop HTTP $code)"
    n=$((n+1))
  done <<< "$stale"
  summary "• [preempt] stopped $n stale e2e workload(s) before this run dispatches"
}

reap_all_stale
