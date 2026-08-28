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
# Stop polling as soon as one leg FAILs. This is a release GATE: the first FAIL already
# blocks the release, so the remaining legs cannot change the verdict -- and waiting them
# out costs the single self-hosted runner, which in turn keeps the next fix's run stuck at
# run-level `pending` (a newer run gets no jobs at all while this one holds the
# concurrency group). Still-running workloads are LEFT ALIVE (see leave_running_wids) so
# they can finish for debugging; only the poll job exits. Set POLL_FAIL_FAST=0 to keep
# polling until every leg reaches a terminal verdict anyway.
POLL_FAIL_FAST="${POLL_FAIL_FAST:-1}"
LEAVE_RUNNING_FILE="${LEAVE_RUNNING_FILE:-${DISPATCH_MAP}.leave_running}"
# Sleep in short slices instead of one long one so a cancelled job tears down in seconds
# rather than at the end of a full POLL_INTERVAL_S.
POLL_SLEEP_SLICE_S="${POLL_SLEEP_SLICE_S:-5}"

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

# ---- sticky PR/commit report comment ---------------------------------------
# This CI runs on pull_request, so the PR number is passed in directly via PR_NUMBER
# (github.event.pull_request.number). If it's absent (e.g. a workflow_dispatch run),
# fall back to resolving the PR from the triggering commit (GET /commits/{sha}/pulls),
# and if that too fails, comment on the commit. Then upsert ONE sticky comment (matched
# by an HTML marker) and PATCH it in place as legs finish -- so a single comment
# updates incrementally (design §10, point C). Mirrors ci-e2e-dispatch.sh.
REPORT_MARKER="<!-- pre-release-e2e-report:${CI_VERSION} -->"
PR_NUMBER="${PR_NUMBER:-}"; COMMENT_TARGET=""   # COMMENT_TARGET: "pr" | "commit" | "" (disabled)

gh_report_on() { [ -n "${GH_STATUS_TOKEN:-}" ] && [ -n "${GH_STATUS_REPO:-}" ] && [ -n "${GH_STATUS_SHA:-}" ]; }

resolve_comment_target() {
  gh_report_on || { COMMENT_TARGET=""; return 0; }
  # Preferred: the pull_request event handed us the PR number directly.
  if [[ "$PR_NUMBER" =~ ^[0-9]+$ ]]; then
    COMMENT_TARGET="pr"; echo "[report] commenting on PR #$PR_NUMBER (from event)"; return 0
  fi
  # Fallback (workflow_dispatch etc.): a commit usually belongs to one PR; take the first.
  PR_NUMBER="$(curl -sS \
    -H "Authorization: Bearer ${GH_STATUS_TOKEN}" \
    -H "Accept: application/vnd.github+json" \
    -H "X-GitHub-Api-Version: 2022-11-28" \
    "${GH_API}/repos/${GH_STATUS_REPO}/commits/${GH_STATUS_SHA}/pulls" 2>/dev/null \
    | jq -r '[.[]|.number][0] // empty' 2>/dev/null || true)"
  if [[ "$PR_NUMBER" =~ ^[0-9]+$ ]]; then
    COMMENT_TARGET="pr"
    echo "[report] commenting on PR #$PR_NUMBER (from commit $GH_STATUS_SHA)"
  else
    COMMENT_TARGET="commit"
    echo "[report] no PR for commit $GH_STATUS_SHA; commenting on the commit"
  fi
}

# GET/PATCH/POST helpers keyed by the marker. The PR path uses the issues API
# (a PR is an issue for comments); the commit path uses the commits API.
_comment_list_url() {
  case "$COMMENT_TARGET" in
    pr)     echo "${GH_API}/repos/${GH_STATUS_REPO}/issues/${PR_NUMBER}/comments?per_page=100" ;;
    commit) echo "${GH_API}/repos/${GH_STATUS_REPO}/commits/${GH_STATUS_SHA}/comments?per_page=100" ;;
  esac
}
_comment_create_url() {
  case "$COMMENT_TARGET" in
    pr)     echo "${GH_API}/repos/${GH_STATUS_REPO}/issues/${PR_NUMBER}/comments" ;;
    commit) echo "${GH_API}/repos/${GH_STATUS_REPO}/commits/${GH_STATUS_SHA}/comments" ;;
  esac
}
_comment_patch_url() { # comment id
  case "$COMMENT_TARGET" in
    pr)     echo "${GH_API}/repos/${GH_STATUS_REPO}/issues/comments/$1" ;;
    commit) echo "${GH_API}/repos/${GH_STATUS_REPO}/comments/$1" ;;
  esac
}

report_upsert() { # body(markdown, already includes the marker on line 1)
  [ -n "$COMMENT_TARGET" ] || return 0
  local body="$1" cid
  cid="$(curl -sS -H "Authorization: Bearer ${GH_STATUS_TOKEN}" -H "Accept: application/vnd.github+json" \
    "$(_comment_list_url)" 2>/dev/null \
    | jq -r --arg m "$REPORT_MARKER" '[.[]|select(.body|contains($m))|.id][0] // empty' 2>/dev/null || true)"
  if [ -n "$cid" ]; then
    curl -sS -X PATCH -H "Authorization: Bearer ${GH_STATUS_TOKEN}" -H "Accept: application/vnd.github+json" \
      "$(_comment_patch_url "$cid")" \
      -d "$(jq -n --arg b "$body" '{body:$b}')" >/dev/null 2>&1 || true
  else
    curl -sS -X POST -H "Authorization: Bearer ${GH_STATUS_TOKEN}" -H "Accept: application/vnd.github+json" \
      "$(_comment_create_url)" \
      -d "$(jq -n --arg b "$body" '{body:$b}')" >/dev/null 2>&1 || true
  fi
}

# Build the sticky report body from the current VERDICT map. `phase` is a short
# status word (Running|Complete) shown in the heading. Legs with no verdict yet
# render as "⏳ pending".
report_body() { # phase  done_count  total_count
  local phase="$1" done="$2" total="$3" leg v vv vd icon rows=""
  for leg in "${LEGS[@]}"; do
    v="${VERDICT[$leg]:-}"
    if [ -z "$v" ]; then
      rows="${rows}| \`${leg}\` | ⏳ pending | running |
"
      continue
    fi
    vv="${v%%|*}"; vd="${v#*|}"
    icon="✅"; [ "$vv" = "PASS" ] || icon="❌"
    rows="${rows}| \`${leg}\` | ${icon} ${vv} | ${vd} |
"
  done
  local detail_link=""
  [ -n "${GH_STATUS_DETAILS_URL:-}" ] && detail_link="[run details](${GH_STATUS_DETAILS_URL})"
  printf '%s\n## Pre-release E2E — %s (%d/%d legs done)\n\nCI_VERSION `%s` · target_gain %s%% · commit `%s`\n\n| leg | verdict | detail |\n|-----|---------|--------|\n%s\n%s\n' \
    "$REPORT_MARKER" "$phase" "$done" "$total" "$CI_VERSION" "$TARGET_GAIN" "$GH_STATUS_SHA" "$rows" "$detail_link"
}

# Count legs that have a terminal verdict.
done_count() {
  local n=0 leg
  for leg in "${LEGS[@]}"; do [ -n "${VERDICT[$leg]:-}" ] && n=$((n+1)); done
  echo "$n"
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

# Resolve the PR (from the triggering commit) or fall back to a commit comment,
# then post the initial "all pending" sticky report.
resolve_comment_target
report_upsert "$(report_body Running "$(done_count)" "${#LEGS[@]}")"

start_s="$(date +%s)"
fail_seen=0   # has any leg reached a FAIL verdict? -> the gate is already decided
while :; do
  pending=0
  changed=0   # did any leg reach a verdict this tick? -> refresh the sticky comment
  for leg in "${LEGS[@]}"; do
    [ -n "${VERDICT[$leg]}" ] && continue
    wid="${WID[$leg]}"
    wphase="$(workload_phase "$wid")"
    # Terminal SaFE failure kills the leg immediately.
    if [ "$wphase" = "Failed" ] || [ "$wphase" = "Stopped" ]; then
      VERDICT["$leg"]="FAIL|workload phase=$wphase"
      summary "❌ **$leg** — FAIL (workload $wphase, wid=\`$wid\`)"
      post_status "$leg" failure "workload $wphase; wid=$wid"
      changed=1; fail_seen=1
      continue
    fi
    # Otherwise judge from the on-disk report (present once the leg finishes).
    res="$(judge_leg "$leg" "$wphase")"
    verdict="${res%%|*}"; detail="${res#*|}"
    if [ "$verdict" = "PASS" ]; then
      VERDICT["$leg"]="PASS|$detail"
      summary "✅ **$leg** — PASS ($detail)"
      post_status "$leg" success "PASS — $detail"
      changed=1
    elif [ "$wphase" = "Succeeded" ]; then
      # Workload ended but the report did not clear the gate -> terminal FAIL.
      VERDICT["$leg"]="FAIL|$detail"
      summary "❌ **$leg** — FAIL ($detail)"
      post_status "$leg" failure "FAIL — $detail"
      changed=1; fail_seen=1
    else
      pending=$((pending + 1))   # still running; check again next tick
    fi
  done

  # A leg finished this tick -> refresh the single sticky report comment (point C).
  [ "$changed" -eq 1 ] && report_upsert "$(report_body Running "$(done_count)" "${#LEGS[@]}")"

  [ "$pending" -eq 0 ] && break

  # Gate already lost -> release the runner, but leave still-running workloads up so they
  # can finish on the cluster (useful for debugging infra vs product failures).
  if [ "$POLL_FAIL_FAST" = "1" ] && [ "$fail_seen" -eq 1 ]; then
    leave_wids=()
    for leg in "${LEGS[@]}"; do
      [ -n "${VERDICT[$leg]}" ] && continue
      VERDICT["$leg"]="SKIP|still running (gate failed; workload left alive)"
      summary "⏳ **$leg** — still running (gate already failed; workload left alive)"
      post_status "$leg" pending "gate failed; workload left running"
      leave_wids+=( "${WID[$leg]}" )
    done
    if [ "${#leave_wids[@]}" -gt 0 ]; then
      printf '%s\n' "${leave_wids[@]}" | sort -u | jq -R . | jq -s . > "$LEAVE_RUNNING_FILE"
      summary ""
      summary "⏹ fail-fast: gate is FAIL. Releasing the runner; ${pending} workload(s) left running for post-mortem. Wids recorded in \`$(basename "$LEAVE_RUNNING_FILE")\`. Set \`POLL_FAIL_FAST=0\` to poll until every leg finishes."
    fi
    break
  fi

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
  slept=0
  while [ "$slept" -lt "$POLL_INTERVAL_S" ]; do
    sleep "$POLL_SLEEP_SLICE_S"
    slept=$(( slept + POLL_SLEEP_SLICE_S ))
  done
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
  case "$vv" in
    PASS) icon="✅" ;;
    SKIP) icon="⏳"; fail=1 ;;
    *) icon="❌"; fail=1 ;;
  esac
  summary "| \`$leg\` | $icon $vv | $vd |"
done
summary ""
if [ "$fail" -eq 0 ]; then
  gate_line="**GATE: PASS** — all ${#LEGS[@]} legs reached target_gain=${TARGET_GAIN}."
else
  gate_line="**GATE: FAIL** — one or more legs did not pass. Release blocked."
fi
summary "$gate_line"

# Final sticky report: the completed table plus the gate verdict.
report_upsert "$(printf '%s\n\n%s\n' "$(report_body Complete "$(done_count)" "${#LEGS[@]}")" "$gate_line")"

# ---- reclaim: STOP (not delete) every workload -----------------------------
# Policy: stop, don't delete. This frees the GPUs immediately (so a 3h leg that
# finished early doesn't idle-hold its card until the SaFE `timeout` deadline) while
# KEEPING the workload record + its pod filesystem for post-hoc inspection. SaFE has
# no `start` endpoint, so these are not resumable; clean up Stopped records manually.
# Verified 2026-08-27: POST /api/v1/workloads/{id}/stop exists and returns 200.
leave_running_wid() { # wid -> 0 if this workload should stay up
  local wid="$1"
  [ -f "$LEAVE_RUNNING_FILE" ] || return 1
  jq -e --arg w "$wid" 'index($w) != null' "$LEAVE_RUNNING_FILE" >/dev/null 2>&1
}

stop_workloads() {
  local wid seen=""
  for leg in "${LEGS[@]}"; do
    wid="${WID[$leg]}"
    # docker legs share one host workload -> stop each unique id once.
    case " $seen " in *" $wid "*) continue ;; esac
    seen="${seen} ${wid}"
    if leave_running_wid "$wid"; then
      summary "• left workload \`$wid\` running (fail-fast; post-mortem)"
      continue
    fi
    code="$(curl -sS "${tls[@]}" -o /dev/null -w '%{http_code}' -X POST \
      "$API/$wid/stop" "${auth[@]}" 2>/dev/null || echo 000)"
    summary "• stopped workload \`$wid\` (HTTP $code)"
  done
}
stop_workloads || true

[ "$fail" -eq 0 ] && exit 0
exit 1
