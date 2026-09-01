#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# Pre-release E2E: poll the dispatched SaFE Authoring workloads and judge each leg
# independently, reporting PASS/FAIL as soon as that leg reaches a terminal state
# (design §10, point C -- the 3h legs finish ~3-4h and report first; 12h legs later).
#
# A leg PASSes only when ALL hold (design §9):
#   1. state.json stop_reason is a clean terminal exit (same set as optimize CLI exit 0),
#   2. crash_count is within tolerance (read from state.json).
# final.json is not required; bootstrap may fail waiting for it while optimize succeeded.
# TARGET_GAIN still flows to optimize via the demo skill; it is NOT used here to judge PASS.
# When the gate is already FAIL, poll keeps running until every leg reaches a terminal
# verdict so per-leg GitHub checks and the sticky report stay aligned with optimize.
# Superseded runs still exit early and leave workloads for dispatch reap.
#
# The exit code is 0 only if every requested leg PASSed.
#
# Requires: bash, curl, jq on the (self-hosted, in-network) runner. state.json is
# written root-only inside pods; poll reads it via sudo -n when needed (passwordless
# sudo on the baremetal runner, same as the NFS-root setup step).
#
# Inputs (env):
#   SAFE_API_BASE / SAFE_API_KEY   SaFE API                       (required)
#   CI_VERSION                     run version                    (required)
#   DISPATCH_MAP                   leg->workloadId JSON from dispatch (required)
#   NFS_ROOT                       (default /shared_nfs/hyperloom-pre-release-e2e-test)
#   TARGET_GAIN                    passed to optimize (demo skill); not used to judge PASS
#   POLL_INTERVAL_S                seconds between polls (default 120)
#   GLOBAL_TIMEOUT_S               hard cap; unfinished legs -> FAIL (default 52200 =
#                                  14.5h, above the 14.25h 12h-leg pod timeout so the
#                                  pod's own death is observed instead of timing out
#                                  first; zombie legs with an empty stop_reason wait
#                                  here rather than a stall check)
#   MAX_CRASHES                    crash_count tolerance (default 0). Server boot
#                                  failures are NOT a pass criterion: the count lives in
#                                  the report journal, not state.json, so nothing here
#                                  reads it -- do not re-add a knob that judges nothing.
#   Optional GitHub commit status (per-leg context pre-release-e2e/<leg>):
#     GH_STATUS_TOKEN / GH_STATUS_REPO / GH_STATUS_SHA / GH_STATUS_DETAILS_URL
#   Supersede detection (release runner when a newer pre-release run is queued):
#     GITHUB_RUN_ID / PR_NUMBER / HEAD_REF (branch); uses GH_STATUS_TOKEN + GH_STATUS_REPO
#   SAFE_CACERT / SAFE_INSECURE    TLS to the API
set -euo pipefail

NFS_ROOT="${NFS_ROOT:-/shared_nfs/hyperloom-pre-release-e2e-test}"
TARGET_GAIN="${TARGET_GAIN:-100}"
POLL_INTERVAL_S="${POLL_INTERVAL_S:-120}"
GLOBAL_TIMEOUT_S="${GLOBAL_TIMEOUT_S:-52200}"
MAX_CRASHES="${MAX_CRASHES:-0}"
LEAVE_RUNNING_FILE="${LEAVE_RUNNING_FILE:-${DISPATCH_MAP}.leave_running}"
API_ERR_FILE="${API_ERR_FILE:-${DISPATCH_MAP}.api_err}"
# Consecutive polls in which EVERY workload query failed. A poll that cannot reach the
# API learns nothing from waiting, so give up rather than hold 8 GPUs and the only
# self-hosted runner until GLOBAL_TIMEOUT_S. Default ~20min at POLL_INTERVAL_S=120.
API_FAIL_ABORT="${API_FAIL_ABORT:-10}"
# This run's dispatch tag, written by dispatch beside the map. Pods stamp it into their
# session pin; an untagged/foreign pin is a leftover from an earlier run on these paths.
RUN_TAG="${RUN_TAG:-$(cat "${DISPATCH_MAP}.version_tag" 2>/dev/null || true)}"
# Sleep in short slices instead of one long one so a cancelled job tears down in seconds
# rather than at the end of a full POLL_INTERVAL_S. Each slice also re-checks whether a
# newer pre-release run has been queued so this poll can exit and release the runner.
POLL_SLEEP_SLICE_S="${POLL_SLEEP_SLICE_S:-5}"
PRE_RELEASE_WORKFLOW_FILE="${PRE_RELEASE_WORKFLOW_FILE:-pre-release-e2e-test.yml}"

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

# True when a newer pre-release workflow run is queued/in-flight for this PR branch.
# GitHub run ids are monotonic; a pending successor blocks on concurrency until we exit.
supersede_check_on() {
  [[ "${GITHUB_RUN_ID:-}" =~ ^[0-9]+$ ]] \
    && [ -n "${GH_STATUS_TOKEN:-}" ] \
    && [ -n "${GH_STATUS_REPO:-}" ]
}

_supersede_head_ref() {
  if [ -n "${HEAD_REF:-}" ]; then
    printf '%s' "$HEAD_REF"
    return 0
  fi
  if [[ "${PR_NUMBER:-}" =~ ^[0-9]+$ ]]; then
    curl -sS \
      -H "Authorization: Bearer ${GH_STATUS_TOKEN}" \
      -H "Accept: application/vnd.github+json" \
      -H "X-GitHub-Api-Version: 2022-11-28" \
      "${GH_API}/repos/${GH_STATUS_REPO}/pulls/${PR_NUMBER}" 2>/dev/null \
      | jq -r '.head.ref // empty' 2>/dev/null || true
    return 0
  fi
  echo ""
}

superseded_by_newer_run() {
  supersede_check_on || return 1
  local head_ref newer
  head_ref="$(_supersede_head_ref)"
  [ -n "$head_ref" ] || return 1
  newer="$(curl -sS \
    -H "Authorization: Bearer ${GH_STATUS_TOKEN}" \
    -H "Accept: application/vnd.github+json" \
    -H "X-GitHub-Api-Version: 2022-11-28" \
    "${GH_API}/repos/${GH_STATUS_REPO}/actions/workflows/${PRE_RELEASE_WORKFLOW_FILE}/runs?branch=${head_ref}&per_page=10" \
    2>/dev/null \
    | jq -r --argjson rid "$GITHUB_RUN_ID" '
        [.workflow_runs[]?
          | select(.id > $rid)
          | select(.status == "queued" or .status == "in_progress" or .status == "pending" or .status == "waiting")
          | .id][0] // empty' 2>/dev/null || true)"
  [ -n "$newer" ]
}

mark_superseded_and_exit_poll() { # -> sets superseded=1, marks pending legs SKIP, breaks caller loop
  local leg pending=0 leave_wids=()
  for leg in "${LEGS[@]}"; do
    [ -n "${VERDICT[$leg]}" ] && continue
    VERDICT["$leg"]="SKIP|superseded by newer run (dispatch reap will stop)"
    summary "⏳ **$leg** — superseded (newer run queued; workload left for dispatch reap)"
    post_status "$leg" pending "superseded; newer run queued"
    leave_wids+=( "${WID[$leg]}" )
    pending=$((pending + 1))
  done
  if [ "${#leave_wids[@]}" -gt 0 ]; then
    printf '%s\n' "${leave_wids[@]}" | sort -u | jq -R . | jq -s . > "$LEAVE_RUNNING_FILE"
  fi
  summary ""
  summary "⏹ superseded: newer pre-release run queued (run_id>${GITHUB_RUN_ID}). Releasing the runner; ${pending} workload(s) left for the successor's dispatch reap."
  superseded=1
}

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

# Icon for a leg verdict in the sticky report table.
verdict_icon() {
  case "$1" in
    PASS) printf '%s' "✅" ;;
    SKIP|PENDING) printf '%s' "⏳" ;;
    *) printf '%s' "❌" ;;
  esac
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
    icon="$(verdict_icon "$vv")"
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

# workloadId -> phase string, or __APIERR__ when the query itself did not answer.
# Discarding curl's stderr here used to make an unreachable or unauthorized API
# indistinguishable from a workload that simply had no phase yet: every leg read
# `Unknown`, which is not terminal, so the poll waited out GLOBAL_TIMEOUT_S with nothing
# in the log to say why. The error text goes to $API_ERR_FILE for the caller to report
# once per tick; this runs inside a command substitution, so it cannot count anything.
workload_phase() {
  local wid="$1" body code
  body="$(curl -sS "${tls[@]}" -w $'\n%{http_code}' "$API/$wid" "${auth[@]}" 2>"$API_ERR_FILE")" || true
  code="$(printf '%s' "$body" | tail -n1)"
  case "$code" in
    2*) printf '%s' "$body" | sed '$d' | jq -r '.phase // "Unknown"' 2>/dev/null || echo Unknown ;;
    *)  printf '%s' "__APIERR__ (HTTP ${code:-none})" ;;
  esac
}

# Resolve a leg's session dir. Bootstrap writes the pinned session dir to
# runs/<CI_VERSION>/<leg>/session/.session_dir (design §9: never guess by timestamp).
leg_session_dir() {
  local leg="$1" pin tag
  pin="${runs_dir}/${leg}/session/.session_dir"
  [ -f "$pin" ] || { echo ""; return; }
  # Line 2 carries the writing run's tag. A pin from an earlier run that reused this
  # CI_VERSION points at a FINISHED session, so honouring it would judge that run's
  # state.json as ours -- a clean stop_reason there would pass the gate before this
  # run's pod has even booted. Treat a foreign or missing tag as "not pinned yet".
  if [ -n "${RUN_TAG:-}" ]; then
    tag="$(sed -n 2p "$pin" 2>/dev/null || true)"
    [ "$tag" = "$RUN_TAG" ] || { echo ""; return; }
  fi
  head -n1 "$pin"
}

# Read one jq filter from state.json. Pods write it root-only (mode 600); the runner
# user reads via sudo -n when needed. Echoes "__UNREADABLE__" when the file exists
# but cannot be opened.
state_json_query() {
  local file="$1" filter="$2"
  if [ ! -f "$file" ]; then
    echo ""; return 0
  fi
  if [ -r "$file" ]; then
    jq -r "$filter" "$file" 2>/dev/null || echo ""
    return 0
  fi
  if sudo -n test -r "$file" 2>/dev/null; then
    sudo -n jq -r "$filter" "$file" 2>/dev/null || echo ""
    return 0
  fi
  echo "__UNREADABLE__"
}

# Clean terminal stop_reason values (hyperloom.inference_optimizer.cli._SUCCESS_STOP_REASONS).
is_clean_stop_reason() {
  case "$1" in
    target_reached|global_converged|time_exhausted|max_ticks|sweep_done|conc_sweep_done)
      return 0 ;;
    *) return 1 ;;
  esac
}

# Judge one leg from state.json. Echoes "PASS"|"PENDING"|"FAIL|<reason>".
judge_leg() {
  local leg="$1" wphase="$2" sdir state gain stop crashes
  sdir="$(leg_session_dir "$leg")"
  if [ -z "$sdir" ] || [ ! -d "$sdir" ]; then
    echo "PENDING|no session dir yet (workload phase=$wphase)"; return
  fi
  state="${sdir%/}/state.json"
  if [ ! -f "$state" ]; then
    echo "PENDING|state.json missing (workload phase=$wphase)"; return
  fi
  stop="$(state_json_query "$state" '.stop_reason // ""')"
  gain="$(state_json_query "$state" '.cumulative_gain_validated // 0')"
  crashes="$(state_json_query "$state" '.crash_count // 0')"
  if [ "$stop" = "__UNREADABLE__" ] || [ "$gain" = "__UNREADABLE__" ] || [ "$crashes" = "__UNREADABLE__" ]; then
    echo "PENDING|state.json not readable (workload phase=$wphase)"; return
  fi
  if [ "$crashes" -gt "$MAX_CRASHES" ] 2>/dev/null; then
    echo "FAIL|crash_count=$crashes > $MAX_CRASHES"; return
  fi
  if [ -z "$stop" ]; then
    echo "PENDING|state.json stop_reason not set yet (workload phase=$wphase)"; return
  fi
  if is_clean_stop_reason "$stop"; then
    echo "PASS|stop=${stop} gain=${gain}%"; return
  fi
  echo "FAIL|stop_reason=${stop} (not a clean terminal exit)"
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
gate_fail_announced=0
superseded=0  # a newer workflow run is queued -> release runner without stopping pods
api_dead=0    # the API has been unreachable for API_FAIL_ABORT consecutive polls
api_fail_streak=0
while :; do
  if superseded_by_newer_run; then
    mark_superseded_and_exit_poll
    break
  fi
  pending=0
  changed=0   # did any leg reach a verdict this tick? -> refresh the sticky comment
  api_err=0; api_queried=0
  for leg in "${LEGS[@]}"; do
    [ -n "${VERDICT[$leg]}" ] && continue
    wid="${WID[$leg]}"
    wphase="$(workload_phase "$wid")"
    api_queried=$(( api_queried + 1 ))
    case "$wphase" in __APIERR__*) api_err=$(( api_err + 1 )) ;; esac
    res="$(judge_leg "$leg" "$wphase")"
    verdict="${res%%|*}"; detail="${res#*|}"
    if [ "$verdict" = "PASS" ]; then
      VERDICT["$leg"]="PASS|$detail"
      summary "✅ **$leg** — PASS ($detail)"
      post_status "$leg" success "PASS — $detail"
      changed=1
    elif [ "$verdict" = "PENDING" ]; then
      if [ "$wphase" = "Succeeded" ] || [ "$wphase" = "Failed" ] || [ "$wphase" = "Stopped" ]; then
        VERDICT["$leg"]="FAIL|$detail"
        summary "❌ **$leg** — FAIL ($detail; workload $wphase, wid=\`$wid\`)"
        post_status "$leg" failure "FAIL — $detail"
        changed=1; fail_seen=1
      else
        pending=$((pending + 1))
      fi
    else
      VERDICT["$leg"]="FAIL|$detail"
      summary "❌ **$leg** — FAIL ($detail)"
      post_status "$leg" failure "FAIL — $detail"
      changed=1; fail_seen=1
    fi
  done

  # Report an API outage once per tick rather than once per leg, and stop waiting on one
  # that is total: state.json still comes off NFS, so a leg that exits cleanly is judged
  # either way -- what an unreachable API costs is the ability to tell a zombie leg from
  # a slow one, which is exactly what the remaining wait would have been for.
  if [ "$api_err" -gt 0 ] && [ "$api_err" -eq "$api_queried" ]; then
    api_fail_streak=$(( api_fail_streak + 1 ))
    api_err_text="$(tr '\r\n' '  ' < "$API_ERR_FILE" 2>/dev/null | head -c 300 || true)"
    echo "WARN: all $api_queried workload queries failed (streak ${api_fail_streak}/${API_FAIL_ABORT}): ${api_err_text:-no stderr from curl}" >&2
    if [ "$api_fail_streak" -ge "$API_FAIL_ABORT" ]; then
      api_dead=1
      summary ""
      summary "❌ **SaFE API unreachable** for ${api_fail_streak} consecutive polls — abandoning the wait instead of holding the runner to the ${GLOBAL_TIMEOUT_S}s timeout."
      summary ""
      summary "\`\`\`"
      summary "${api_err_text:-no stderr from curl}"
      summary "\`\`\`"
      for leg in "${LEGS[@]}"; do
        [ -n "${VERDICT[$leg]}" ] && continue
        VERDICT["$leg"]="FAIL|SaFE API unreachable for ${api_fail_streak} polls"
        post_status "$leg" error "SaFE API unreachable"
      done
      break
    fi
  else
    [ "$api_fail_streak" -gt 0 ] && echo "[poll] workload queries recovered after ${api_fail_streak} failed poll(s)"
    api_fail_streak=0
  fi

  # A leg finished this tick -> refresh the single sticky report comment (point C).
  [ "$changed" -eq 1 ] && report_upsert "$(report_body Running "$(done_count)" "${#LEGS[@]}")"

  if [ "$fail_seen" -eq 1 ] && [ "$gate_fail_announced" -eq 0 ]; then
    summary ""
    summary "⚠️ **GATE: FAIL** — at least one leg did not pass. Continuing to poll until every leg reaches a terminal verdict."
    gate_fail_announced=1
  fi

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
  slept=0
  while [ "$slept" -lt "$POLL_INTERVAL_S" ]; do
    if superseded_by_newer_run; then
      mark_superseded_and_exit_poll
      break
    fi
    sleep "$POLL_SLEEP_SLICE_S"
    slept=$(( slept + POLL_SLEEP_SLICE_S ))
  done
  [ "$superseded" -eq 1 ] && break
done

if [ "$superseded" -eq 1 ]; then
  summary ""
  summary "### Result"
  summary ""
  summary "| leg | verdict | detail |"
  summary "|-----|---------|--------|"
  for leg in "${LEGS[@]}"; do
    v="${VERDICT[$leg]:-SKIP|superseded}"
    vv="${v%%|*}"; vd="${v#*|}"
    summary "| \`$leg\` | ⏳ $vv | $vd |"
  done
  summary ""
  gate_line="**GATE: SUPERSEDED** — newer run queued; workloads left for dispatch reap."
  summary "$gate_line"
  report_upsert "$(printf '%s\n\n%s\n' "$(report_body Complete "$(done_count)" "${#LEGS[@]}")" "$gate_line")"
  exit 0
fi

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
  gate_line="**GATE: PASS** — all ${#LEGS[@]} legs completed with a clean terminal stop_reason."
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
      summary "• left workload \`$wid\` running (superseded; dispatch reap)"
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
