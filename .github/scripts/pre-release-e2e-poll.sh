#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# Pre-release E2E: poll the dispatched SaFE Authoring workloads and judge each leg
# independently, reporting PASS/FAIL as soon as that leg reaches a terminal state
# (design §10, point C -- the 3h legs finish ~3-4h and report first; 12h legs later).
#
# A leg PASSes only when ALL hold (design §9):
#   1. its session reports/final.json has stop_reason == "target_reached"
#      (equivalently cumulative_gain_validated >= TARGET_GAIN; the gate is 100),
#   2. its owning SaFE workload phase is not Failed/Stopped,
#   3. reports/final.json and reports/final.md both exist,
#   4. crash_count / server_boot_failures are within tolerance.
# Anything else (incl. "ran the full duration without target_reached") is FAIL.
#
# The exit code is 0 only if every requested leg PASSed.
#
# Requires: bash, curl, jq on the (self-hosted, in-network) runner with the NFS
# runs/ tree readable.
#
# Inputs (env):
#   SAFE_API_BASE / SAFE_API_KEY   SaFE API                       (required)
#   CI_VERSION                     run version                    (required)
#   DISPATCH_MAP                   leg->workloadId JSON from dispatch (required)
#   NFS_ROOT                       (default /shared_nfs/hyperloom-pre-release-e2e-test)
#   TARGET_GAIN                    gate %% (default 100)
#   POLL_INTERVAL_S                seconds between polls (default 120)
#   GLOBAL_TIMEOUT_S               hard cap; unfinished legs -> FAIL
#                                  (default 50400 = 14h)
#   MAX_CRASHES / MAX_BOOT_FAILS   tolerance (default 0 / 0)
#   Optional GitHub commit status (per-leg context pre-release-e2e/<leg>):
#     GH_STATUS_TOKEN / GH_STATUS_REPO / GH_STATUS_SHA / GH_STATUS_DETAILS_URL
#   SAFE_CACERT / SAFE_INSECURE    TLS to the API
set -euo pipefail

NFS_ROOT="${NFS_ROOT:-/shared_nfs/hyperloom-pre-release-e2e-test}"
TARGET_GAIN="${TARGET_GAIN:-100}"
POLL_INTERVAL_S="${POLL_INTERVAL_S:-120}"
GLOBAL_TIMEOUT_S="${GLOBAL_TIMEOUT_S:-50400}"
MAX_CRASHES="${MAX_CRASHES:-0}"
MAX_BOOT_FAILS="${MAX_BOOT_FAILS:-0}"

: "${SAFE_API_BASE:?SAFE_API_BASE is required}"
: "${SAFE_API_KEY:?SAFE_API_KEY is required}"
: "${CI_VERSION:?CI_VERSION is required}"
: "${DISPATCH_MAP:?DISPATCH_MAP is required}"
[ -f "$DISPATCH_MAP" ] || { echo "dispatch map $DISPATCH_MAP not found" >&2; exit 2; }

API="${SAFE_API_BASE%/}/api/v1/workloads"
auth=(-H "Authorization: Bearer ${SAFE_API_KEY}")
tls=()
if [ -n "${SAFE_CACERT:-}" ]; then
  tls=(--cacert "$SAFE_CACERT")
elif [ "${SAFE_INSECURE:-0}" = "1" ]; then
  tls=(-k)
fi
GH_API="${GH_API:-https://api.github.com}"

runs_dir="${NFS_ROOT%/}/runs/${CI_VERSION}"
summary() { echo "$*" | tee -a "${GITHUB_STEP_SUMMARY:-/dev/null}"; }

gh_status_on() { [ -n "${GH_STATUS_TOKEN:-}" ] && [ -n "${GH_STATUS_REPO:-}" ] && [ -n "${GH_STATUS_SHA:-}" ]; }
post_status() { # leg state(pending|success|failure|error) description
  gh_status_on || return 0
  local desc="${3:0:139}"
  curl -sS -X POST \
    -H "Authorization: Bearer ${GH_STATUS_TOKEN}" \
    -H "Accept: application/vnd.github+json" \
    -H "X-GitHub-Api-Version: 2022-11-28" \
    "${GH_API}/repos/${GH_STATUS_REPO}/statuses/${GH_STATUS_SHA}" \
    -d "$(jq -n --arg s "$2" --arg d "$desc" --arg u "${GH_STATUS_DETAILS_URL:-}" --arg c "pre-release-e2e/$1" \
        '{state:$s, description:$d, context:$c} + (if $u=="" then {} else {target_url:$u} end)')" \
    >/dev/null 2>&1 || true
}

workload_phase() { # workloadId -> phase string
  local wid="$1" detail
  detail="$(curl -sS "${tls[@]}" "$API/$wid" "${auth[@]}" 2>/dev/null || true)"
  printf '%s' "$detail" | jq -r '.phase // "Unknown"' 2>/dev/null || echo Unknown
}

# Resolve a leg's session dir. Bootstrap writes the pinned session dir to
# runs/<CI_VERSION>/<leg>/session/.session_dir (design §9: never guess by timestamp).
leg_session_dir() {
  local leg="$1" pin
  pin="${runs_dir}/${leg}/session/.session_dir"
  if [ -f "$pin" ]; then head -n1 "$pin"; return; fi
  echo ""
}

# Judge one leg from its final.json. Echoes "PASS"|"FAIL|<reason>".
judge_leg() {
  local leg="$1" wphase="$2" sdir final gain stop crashes boots
  sdir="$(leg_session_dir "$leg")"
  if [ -z "$sdir" ] || [ ! -d "$sdir" ]; then
    echo "FAIL|no session dir yet (workload phase=$wphase)"; return
  fi
  final="${sdir%/}/reports/final.json"
  if [ ! -f "$final" ]; then
    echo "FAIL|reports/final.json missing (workload phase=$wphase)"; return
  fi
  if [ ! -f "${sdir%/}/reports/final.md" ]; then
    echo "FAIL|reports/final.md missing"; return
  fi
  stop="$(jq -r '.stop_reason // ""' "$final" 2>/dev/null || echo "")"
  gain="$(jq -r '.cumulative_gain_validated // 0' "$final" 2>/dev/null || echo 0)"
  crashes="$(jq -r '.crash_count // 0' "$final" 2>/dev/null || echo 0)"
  boots="$(jq -r '.server_boot_failures // 0' "$final" 2>/dev/null || echo 0)"
  if [ "$crashes" -gt "$MAX_CRASHES" ] 2>/dev/null; then
    echo "FAIL|crash_count=$crashes > $MAX_CRASHES"; return
  fi
  if [ "$boots" -gt "$MAX_BOOT_FAILS" ] 2>/dev/null; then
    echo "FAIL|server_boot_failures=$boots > $MAX_BOOT_FAILS"; return
  fi
  # Primary gate: stop_reason target_reached (== gain >= TARGET_GAIN).
  if [ "$stop" = "target_reached" ]; then
    echo "PASS|gain=${gain}% stop=${stop}"; return
  fi
  # Fallback: numeric compare in case stop_reason lags (awk for float).
  if awk -v g="$gain" -v t="$TARGET_GAIN" 'BEGIN{exit !(g+0 >= t+0)}'; then
    echo "PASS|gain=${gain}% (>= ${TARGET_GAIN})"; return
  fi
  echo "FAIL|gain=${gain}% < ${TARGET_GAIN} (stop=${stop:-none})"
}

# ---- poll loop -------------------------------------------------------------
mapfile -t LEGS < <(jq -r 'keys[]' "$DISPATCH_MAP")
declare -A WID VERDICT
for leg in "${LEGS[@]}"; do
  WID["$leg"]="$(jq -r --arg l "$leg" '.[$l]' "$DISPATCH_MAP")"
  VERDICT["$leg"]=""
  post_status "$leg" pending "dispatched; workload=${WID[$leg]}"
done
summary "## Pre-release E2E — CI_VERSION \`$CI_VERSION\`"
summary ""
summary "Polling ${#LEGS[@]} legs (global timeout $((GLOBAL_TIMEOUT_S/3600))h). Each leg reports on its own terminal (point C)."
summary ""

start_s="$(date +%s)"
while :; do
  pending=0
  for leg in "${LEGS[@]}"; do
    [ -n "${VERDICT[$leg]}" ] && continue
    wid="${WID[$leg]}"
    wphase="$(workload_phase "$wid")"
    # Terminal SaFE failure kills the leg immediately.
    if [ "$wphase" = "Failed" ] || [ "$wphase" = "Stopped" ]; then
      VERDICT["$leg"]="FAIL|workload phase=$wphase"
      summary "❌ **$leg** — FAIL (workload $wphase, wid=\`$wid\`)"
      post_status "$leg" failure "workload $wphase; wid=$wid"
      continue
    fi
    # Otherwise judge from the on-disk report (present once the leg finishes).
    res="$(judge_leg "$leg" "$wphase")"
    verdict="${res%%|*}"; detail="${res#*|}"
    if [ "$verdict" = "PASS" ]; then
      VERDICT["$leg"]="PASS|$detail"
      summary "✅ **$leg** — PASS ($detail)"
      post_status "$leg" success "PASS — $detail"
    elif [ "$wphase" = "Succeeded" ]; then
      # Workload ended but the report did not clear the gate -> terminal FAIL.
      VERDICT["$leg"]="FAIL|$detail"
      summary "❌ **$leg** — FAIL ($detail)"
      post_status "$leg" failure "FAIL — $detail"
    else
      pending=$((pending + 1))   # still running; check again next tick
    fi
  done

  [ "$pending" -eq 0 ] && break

  elapsed=$(( $(date +%s) - start_s ))
  if [ "$elapsed" -ge "$GLOBAL_TIMEOUT_S" ]; then
    for leg in "${LEGS[@]}"; do
      [ -n "${VERDICT[$leg]}" ] && continue
      VERDICT["$leg"]="FAIL|global timeout after ${elapsed}s"
      summary "❌ **$leg** — FAIL (global timeout)"
      post_status "$leg" failure "global timeout after $((elapsed/3600))h"
    done
    break
  fi
  echo "[poll] ${pending} leg(s) still running; elapsed $((elapsed/60))m; sleeping ${POLL_INTERVAL_S}s"
  sleep "$POLL_INTERVAL_S"
done

# ---- aggregate gate --------------------------------------------------------
summary ""
summary "### Result"
summary ""
summary "| leg | verdict | detail |"
summary "|-----|---------|--------|"
fail=0
for leg in "${LEGS[@]}"; do
  v="${VERDICT[$leg]:-FAIL|no verdict}"
  vv="${v%%|*}"; vd="${v#*|}"
  icon="✅"; [ "$vv" = "PASS" ] || { icon="❌"; fail=1; }
  summary "| \`$leg\` | $icon $vv | $vd |"
done
summary ""
if [ "$fail" -eq 0 ]; then
  summary "**GATE: PASS** — all ${#LEGS[@]} legs reached target_gain=${TARGET_GAIN}."
  exit 0
fi
summary "**GATE: FAIL** — one or more legs did not pass. Release blocked."
exit 1
