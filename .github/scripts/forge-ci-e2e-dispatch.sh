#!/usr/bin/env bash
# Hyperloom Forge CI E2E: submit one single-GPU ``kernelforge forge-loop``
# smoke run built from the exact Hyperloom PR commit, then poll it to terminal.
#
# This is the successor to KernelForge's standalone ci-e2e-dispatch.sh. The
# registered ``kernelforge`` workload template still owns GPU bootstrap and the
# actual Triton-softmax campaign; this script supplies the vendored Hyperloom
# source tree and maps the workload result back to the PR.
set -euo pipefail

GPUS="${GPUS:-1}"
MAX_HOURS="${MAX_HOURS:-1.0}"
MAX_ITERS="${MAX_ITERS:-100}"
GPU_TARGET="${GPU_TARGET:-gfx950}"
POLL_INTERVAL_S="${POLL_INTERVAL_S:-60}"

: "${DISPATRON_BASE_URL:?DISPATRON_BASE_URL is required}"
: "${HEAD_REF:?HEAD_REF (PR head branch) is required}"
: "${HEAD_SHA:?HEAD_SHA (immutable PR head commit) is required}"

# The infra-type check is gone with the endpoint that honoured it. Dispatron places a
# run from its platform row, so "which infrastructure" is not the caller's to assert.

# Checked by name rather than left to fail as `command not found`, which reads like a
# broken workflow rather than a job that landed on a runner without the CLI.
command -v dispatron-ci >/dev/null 2>&1 || {
  echo "::error::dispatron-ci is not on PATH; this runner was not built from Dispatron's deploy/ci/runner.Dockerfile" >&2
  exit 2
}

sanitize_repo_url() {
  # API-visible workload params must never contain the GitHub token.
  printf '%s' "${1:-}" | sed -E 's#https://[^@/]+@#https://#'
}

SRC_REPO="$(sanitize_repo_url "${BASE_REPO_URL:-}")"
PULL_REF="${PR_PULL_REF:-}"
if [ -z "$PULL_REF" ] && [[ "${PR_NUMBER:-}" =~ ^[0-9]+$ ]]; then
  PULL_REF="refs/pull/${PR_NUMBER}/head"
fi
PR_CHECK_BASE="${CI_E2E_PR_CHECK_BASE:-/tmp/ci-e2e}"
SRC_DIR="${CI_E2E_SOURCE_DIR:-${PR_CHECK_BASE%/}/pr_${PR_NUMBER:-manual}/${HEAD_SHA}/hyperloom}"
WORKSPACE="${CI_E2E_WORKSPACE:-control-plan-hyperloom-ci}"

# Bootstrap happens before forge-loop starts counting MAX_HOURS. Keep the
# server-side deadline and the poll window derived from the same budget.
BOOTSTRAP_SLACK_SEC="${CI_E2E_BOOTSTRAP_SLACK_SEC:-3600}"
DEADLINE_SEC="${CI_E2E_DEADLINE_SEC:-$(awk -v h="$MAX_HOURS" -v s="$BOOTSTRAP_SLACK_SEC" \
  'BEGIN{printf "%d", h*3600 + s}')}"
POLL_MAX="${POLL_MAX:-$(awk -v d="$DEADLINE_SEC" -v i="$POLL_INTERVAL_S" \
  'BEGIN{printf "%d", int((d + i - 1) / i) + 5}')}"
IMAGE="${CI_E2E_IMAGE:-harbor.crusoe.primus-safe.amd.com/proxy/vllm/vllm-openai-rocm:v0.24.0}"

summary() { echo "$*" | tee -a "${GITHUB_STEP_SUMMARY:-/dev/null}"; }

STATUS_INTERVAL_S="${STATUS_INTERVAL_S:-300}"
STATUS_CONTEXT="${STATUS_CONTEXT:-ci-e2e/kernelforge}"
GH_API="${GH_API:-https://api.github.com}"
TERMINAL_MARKER="${E2E_TERMINAL_MARKER:-${RUNNER_TEMP:-/tmp}/forge_e2e_status_terminal}"
rm -f "$TERMINAL_MARKER" 2>/dev/null || true

gh_status_on() {
  [ -n "${GH_STATUS_TOKEN:-}" ] && [ -n "${GH_STATUS_REPO:-}" ] && [ -n "${GH_STATUS_SHA:-}" ]
}

post_status() { # state description
  gh_status_on || return 0
  local desc="${2:0:139}" code
  code="$(curl -sS -o /dev/null -w '%{http_code}' -X POST \
    -H "Authorization: Bearer ${GH_STATUS_TOKEN}" \
    -H "Accept: application/vnd.github+json" \
    -H "X-GitHub-Api-Version: 2022-11-28" \
    "${GH_API}/repos/${GH_STATUS_REPO}/statuses/${GH_STATUS_SHA}" \
    -d "$(jq -n --arg s "$1" --arg d "$desc" --arg u "${GH_STATUS_DETAILS_URL:-}" --arg c "$STATUS_CONTEXT" \
      '{state:$s, description:$d, context:$c} + (if $u=="" then {} else {target_url:$u} end)')" \
    2>/dev/null || echo 000)"
  if [ "$code" -ge 200 ] && [ "$code" -lt 300 ]; then
    [ "$1" != "pending" ] && { : > "$TERMINAL_MARKER"; } 2>/dev/null || true
  else
    echo "[forge-ci-e2e] WARN: commit status '$1' not accepted (HTTP $code)" >&2
  fi
}

REPORT_MARKER="<!-- hyperloom-forge-ci-e2e-report -->"
gh_report_on() {
  [ -n "${GH_STATUS_TOKEN:-}" ] && [ -n "${GH_STATUS_REPO:-}" ] && [[ "${PR_NUMBER:-}" =~ ^[0-9]+$ ]]
}

# `fetch_forge_result` was here. It read the run's stdout back through the retired
# backend's log-search route -- GET /workloads/{uid}/logs?keywords=__FORGE_RESULT__ --
# and Dispatron has no such route. The renderer already guards the whole performance
# block on having a result, so `forge_result` stays empty and the comment loses the
# baseline/best/speedup/validation rows.
#
# A real loss and a small one: those rows have not been produced for some time. The
# fetch runs only on the Succeeded branch, and the last hundred workflow runs contain
# no successful GPU job; further back, a green run's log carries no __FORGE_RESULT__
# either, and nobody has said the numbers went missing. Restoring them needs a result
# channel on the run record rather than a log scrape -- see Dispatron's TODO.md,
# "A run's own result has nowhere to live".

report_upsert() { # result
  gh_report_on || return 0
  local result="$1" body cid detail_file forge_result_file
  detail_file="$(mktemp)"
  forge_result_file="$(mktemp)"
  printf '%s' "${detail:-}" > "$detail_file"
  printf '%s' "${forge_result:-}" > "$forge_result_file"
  if ! body="$(python3 .github/scripts/forge_e2e_report.py render \
      --result-label "$result" \
      --detail-file "$detail_file" \
      --forge-result-file "$forge_result_file" \
      --max-hours "$MAX_HOURS" \
      --max-iters "$MAX_ITERS" \
      --gpus "$GPUS" \
      --workspace "$WORKSPACE" \
      --head-ref "$HEAD_REF" \
      --head-sha "$HEAD_SHA" \
      --session-id "$UID_" \
      --details-url "${GH_STATUS_DETAILS_URL:-}" \
      --error "${err:-}" 2>/dev/null)"; then
    echo "[forge-ci-e2e] WARN: rich report rendering failed; posting the minimal report" >&2
    body="${REPORT_MARKER}
## Hyperloom Forge E2E — ${result}

| item | value |
|---|---|
| example | \`triton-softmax-forge-loop\` |
| budget | ${MAX_HOURS} h |
| resources | ${GPUS}× GPU |
| PR branch | \`${HEAD_REF}\` |
| commit | \`${HEAD_SHA}\` |
| session_id | \`${UID_}\` |"
  fi
  rm -f "$detail_file" "$forge_result_file"
  cid="$(curl -sS -H "Authorization: Bearer ${GH_STATUS_TOKEN}" -H "Accept: application/vnd.github+json" \
    "${GH_API}/repos/${GH_STATUS_REPO}/issues/${PR_NUMBER}/comments?per_page=100" 2>/dev/null \
    | jq -r --arg m "$REPORT_MARKER" '[.[]|select(.body|contains($m))|.id][0] // empty' 2>/dev/null || true)"
  if [ -n "$cid" ]; then
    curl -sS -X PATCH -H "Authorization: Bearer ${GH_STATUS_TOKEN}" -H "Accept: application/vnd.github+json" \
      "${GH_API}/repos/${GH_STATUS_REPO}/issues/comments/${cid}" \
      -d "$(jq -n --arg b "$body" '{body:$b}')" >/dev/null 2>&1 || true
  else
    curl -sS -X POST -H "Authorization: Bearer ${GH_STATUS_TOKEN}" -H "Accept: application/vnd.github+json" \
      "${GH_API}/repos/${GH_STATUS_REPO}/issues/${PR_NUMBER}/comments" \
      -d "$(jq -n --arg b "$body" '{body:$b}')" >/dev/null 2>&1 || true
  fi
}

# ---- run it ---------------------------------------------------------------
# `dispatron-ci` submits and polls. The events file is one JSON object per poll, which
# the live status refreshes from; the terminal event carries the verdict, the reason and
# the timings. Neither is this script parsing the CLI's prose.
WORK="$(mktemp -d)"
EVENTS="$WORK/events.jsonl"
# The CLI writes step outputs of its own. Pointed at a scratch file rather than the
# job's, so what this step exposes stays this script's contract and not a union of two.
OUTPUTS="$WORK/cli-outputs"
: > "$EVENTS"; : > "$OUTPUTS"
trap 'rm -rf "$WORK"' EXIT

insecure=()
[ "${CI_E2E_INSECURE:-0}" = "1" ] && insecure=(--insecure)

post_status "pending" "dispatching; sha=${HEAD_SHA:0:12}"

GITHUB_OUTPUT="$OUTPUTS" \
DISPATCH_EVENTS_FILE="$EVENTS" \
dispatron-ci \
  --base-url "$DISPATRON_BASE_URL" \
  --kind kernelforge \
  --image "$IMAGE" \
  --name "forge-ci-pr-${PR_NUMBER:-manual}-${HEAD_SHA:0:12}" \
  --source github-hyperloom-forge-ci \
  --gpus "$GPUS" \
  --max-hours "$MAX_HOURS" \
  --user "${CI_E2E_USER_NAME:-}" \
  --ref "$HEAD_REF" \
  --sha "$HEAD_SHA" \
  --source-repo "$SRC_REPO" \
  --source-dir "$SRC_DIR" \
  --pull-ref "$PULL_REF" \
  --pr "${PR_NUMBER:-}" \
  --poll-interval "$POLL_INTERVAL_S" \
  --poll-max "$POLL_MAX" \
  "${insecure[@]}" &
cli=$!

# Refresh the commit status while the run is going, from the events file rather than
# from a second poll of the API -- a status refresh must not add load to the thing it is
# describing, nor disagree with it.
last_push=0
while kill -0 "$cli" 2>/dev/null; do
  now_s="$(date +%s)"
  if [ $((now_s - last_push)) -ge "$STATUS_INTERVAL_S" ]; then
    ev="$(jq -c 'select(.event=="phase")' "$EVENTS" 2>/dev/null | tail -1 || true)"
    if [ -n "$ev" ]; then
      ph="$(printf '%s' "$ev" | jq -r '.phase // "?"' 2>/dev/null || echo '?')"
      jr="$(printf '%s' "$ev" | jq -r '.platform_ref // ""' 2>/dev/null || true)"
      post_status "pending" "running ${ph}; job=${jr:--}; sha=${HEAD_SHA:0:12}"
    fi
    last_push="$now_s"
  fi
  sleep 5
done
rc=0; wait "$cli" || rc=$?

# Read what the CLI decided, from the terminal event rather than the step-output file.
# Both carry the same fields, but the event is JSON: a platform error containing a
# newline needs no delimiter convention to survive.
term="$(jq -c 'select(.event=="terminal")' "$EVENTS" 2>/dev/null | tail -1 || true)"
field() { [ -n "$term" ] && printf '%s' "$term" | jq -r --arg k "$1" '.[$k] // ""' 2>/dev/null || true; }

UID_="$(field uid)"
result="$(field result)"
err="$(field reason)"
explanation="$(field explanation)"
jobref="$(field platform_ref)"
node="$(field nodes)"
[ -z "$UID_" ] && UID_="$(jq -r 'select(.event=="submitted")|.uid' "$EVENTS" 2>/dev/null | tail -1 || true)"

# The CLI died before writing an outcome -- a refused submit, or the facade unreachable.
# Not a run that failed, and saying so stops somebody debugging code that never ran.
[ -z "$result" ] && result="dispatch-error"

# The performance rows the renderer would add come from a result this path cannot fetch.
forge_result=""

# The renderer reads its timeline out of `detail.orchestration.conditions[]`, which used
# to be the raw status body. The same three instants come back on the terminal event, so
# they are put back into the shape it already reads rather than changing it -- otherwise
# queue-to-dispatch, run time and total all render as "–" for want of a wrapper.
detail="$(jq -nc --arg q "$(field queued_at)" --arg d "$(field dispatched_at)" \
  --arg e "$(field ended_at)" --arg ph "$(field phase)" \
  '{orchestration:{conditions:
     ([{phase:"Queued",time:$q}]
      + (if $d == "" then [] else [{phase:"Dispatched",time:$d}] end)
      + (if $e == "" then [] else [{phase:$ph,time:$e}] end))}}' 2>/dev/null || echo '{}')"

if [ -n "${GITHUB_OUTPUT:-}" ] && [ "${GITHUB_OUTPUT}" != "$OUTPUTS" ]; then
  {
    echo "session_id=${UID_}"
    echo "result=${result}"
    echo "platform_ref=${jobref}"
  } >> "$GITHUB_OUTPUT"
fi

case "$result" in
  succeeded)
    summary "✅ **PASS** — forge-loop smoke completed. session_id=\`${UID_}\` job=\`${jobref:--}\`"
    post_status "success" "PASS — uid=${UID_}; job=${jobref:--}; sha=${HEAD_SHA:0:12}"
    report_upsert "✅ Succeeded" ;;
  cancelled)
    # Not a red build: a newer commit or a retest ended this one, and reporting it as a
    # failure sends somebody looking for a bug that is not there.
    summary "🚫 **CANCELLED** — session_id=\`${UID_}\`"
    post_status "error" "cancelled; uid=${UID_}; sha=${HEAD_SHA:0:12}"
    report_upsert "🚫 Cancelled" ;;
  timeout)
    summary "❌ **FAIL (timeout)** — ${explanation:-gave up waiting}. session_id=\`${UID_}\`"
    post_status "failure" "timeout; uid=${UID_}; sha=${HEAD_SHA:0:12}"
    report_upsert "⏱ Timed out" ;;
  dispatch-error)
    summary "❌ **FAIL** — the run was never dispatched; see the job log."
    post_status "error" "could not dispatch; sha=${HEAD_SHA:0:12}"
    report_upsert "❌ Not dispatched" ;;
  *)
    summary "❌ **FAIL** — session_id=\`${UID_}\` job=\`${jobref:--}\` node=\`${node:--}\`"
    summary "reason: ${explanation:-unknown}"
    post_status "failure" "FAIL (${HEAD_SHA:0:12}): ${explanation:0:110}"
    report_upsert "❌ Failed" ;;
esac

exit "$rc"
