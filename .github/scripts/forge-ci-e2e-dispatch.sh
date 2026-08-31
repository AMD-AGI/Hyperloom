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

: "${E2E_API_BASE:?E2E_API_BASE is required}"
: "${E2E_API_KEY:?E2E_API_KEY is required}"
: "${E2E_INFRA_TYPE:?E2E_INFRA_TYPE is required}"
: "${HEAD_REF:?HEAD_REF (PR head branch) is required}"
: "${HEAD_SHA:?HEAD_SHA (immutable PR head commit) is required}"
if [ "$E2E_INFRA_TYPE" != "kubernetes" ]; then
  echo "Forge E2E supports only E2E_INFRA_TYPE=kubernetes; got '$E2E_INFRA_TYPE'" >&2
  exit 2
fi

API_PREFIX="${CI_E2E_API_PREFIX:-/api/v1}"
API="${E2E_API_BASE%/}${API_PREFIX}/orchestration/workloads"
LOG_API="${E2E_API_BASE%/}${API_PREFIX}/workloads"

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
auth=(-H "Authorization: Bearer ${E2E_API_KEY}")
tls=()
if [ -n "${CI_E2E_CACERT:-}" ]; then
  tls=(--cacert "$CI_E2E_CACERT")
elif [ "${CI_E2E_INSECURE:-0}" = "1" ]; then
  tls=(-k)
fi

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

fetch_forge_result() {
  local payload result attempt
  # Log ingestion can trail the terminal workload phase briefly. The report is
  # best-effort and must never turn a successful GPU run red.
  for attempt in 1 2 3 4 5 6; do
    payload="$(curl -fsS "${tls[@]}" \
      "$LOG_API/$UID_/logs?keywords=__FORGE_RESULT__&tail=50&since=2h" \
      "${auth[@]}" 2>/dev/null || true)"
    result="$(printf '%s' "$payload" | python3 .github/scripts/forge_e2e_report.py extract 2>/dev/null || true)"
    if [ -n "$result" ]; then
      printf '%s' "$result"
      return 0
    fi
    [ "$attempt" -lt 6 ] && sleep 5
  done
  return 1
}

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

params="$(jq -n \
  --arg ref "$HEAD_REF" --arg sha "$HEAD_SHA" --arg srcrepo "$SRC_REPO" --arg srcdir "$SRC_DIR" \
  --arg task_source "github-hyperloom-forge-ci" --arg pr_number "${PR_NUMBER:-}" \
  --arg pullref "$PULL_REF" --arg deadline "$DEADLINE_SEC" \
  --arg max_hours "$MAX_HOURS" --arg max_iters "$MAX_ITERS" --arg gpu_target "$GPU_TARGET" \
  '{KERNELFORGE_SOURCE_REF:$ref, KERNELFORGE_SOURCE_SHA:$sha,
    KERNELFORGE_SOURCE_REPO:$srcrepo, KF_SOURCE_DIR:$srcdir,
    KERNELFORGE_SOURCE_PULL_REF:$pullref,
    KF_TASK_SOURCE:$task_source, KF_SOURCE_PR:$pr_number,
    KF_USE_GIT:"1", KF_ACTIVE_DEADLINE_SEC:$deadline,
    MAX_HOURS:$max_hours, MAX_ITERS:$max_iters, GPU_TARGET:$gpu_target}')"
body="$(jq -n \
  --arg name "forge-ci-pr-${PR_NUMBER:-manual}-${GITHUB_RUN_ID:-local}" \
  --arg uname "${CI_E2E_USER_NAME:-}" --arg itype "$E2E_INFRA_TYPE" \
  --arg ws "$WORKSPACE" --arg img "$IMAGE" --arg git_token "${GITHUB_TOKEN:-}" \
  --arg forge_model "${FORGE_MODEL:-}" \
  --argjson gpus "$GPUS" --argjson params "$params" \
  '{name:$name, infra_type:$itype, kind:"kernelforge", replicas:1,
    gpu_per_replica:$gpus, namespace:$ws, workspace:$ws, image:$img,
    template:{params:$params,
      env:((if $git_token != "" then {GITHUB_TOKEN:$git_token} else {} end)
           + (if $forge_model != "" then {FORGE_MODEL:$forge_model} else {} end))}}
   + (if $uname == "" then {} else {user_name:$uname} end)')"

echo "[forge-ci-e2e] submitting: ref=$HEAD_REF sha=$HEAD_SHA gpus=$GPUS max_hours=$MAX_HOURS" \
  "deadline=${DEADLINE_SEC}s poll=${POLL_MAX}x${POLL_INTERVAL_S}s workspace=$WORKSPACE"
resp="$(curl -sS "${tls[@]}" -w $'\n%{http_code}' -X POST "$API" \
  "${auth[@]}" -H "Content-Type: application/json" -d "$body")"
code="$(printf '%s' "$resp" | tail -n1)"
json="$(printf '%s' "$resp" | sed '$d')"
if [ "$code" -lt 200 ] || [ "$code" -ge 300 ]; then
  summary "❌ submit failed (HTTP $code): $(printf '%s' "$json" | head -c 500)"
  exit 1
fi

UID_="$(printf '%s' "$json" | jq -r '.uid // empty')"
if [ -z "$UID_" ]; then
  summary "❌ submit returned no uid: $(printf '%s' "$json" | head -c 500)"
  exit 1
fi
summary "**session_id (workload uid):** \`$UID_\`"
echo "session_id=$UID_" >> "${GITHUB_OUTPUT:-/dev/null}"
echo "$UID_" > "${E2E_UID_FILE:-${RUNNER_TEMP:-/tmp}/forge_e2e_session_uid}" 2>/dev/null || true

post_status "pending" "submitted; uid=${UID_}; ref=${HEAD_REF}; sha=${HEAD_SHA:0:12}"
last_push="$(date +%s)"
cleanup() { curl -sS "${tls[@]}" -X DELETE "$API/$UID_" "${auth[@]}" >/dev/null 2>&1 || true; }
trap 'echo "[forge-ci-e2e] cancelled; deleting workload $UID_"; post_status "error" "cancelled; uid=${UID_}; sha=${HEAD_SHA:0:12}"; cleanup; exit 1' INT TERM

i=0
prev_phase=""
while [ "$i" -lt "$POLL_MAX" ]; do
  i=$((i + 1))
  detail="$(curl -sS "${tls[@]}" "$API/$UID_" "${auth[@]}" || true)"
  phase="$(printf '%s' "$detail" | jq -r '.orchestration.phase // "Unknown"' 2>/dev/null || echo Unknown)"
  jobref="$(printf '%s' "$detail" | jq -r '.dispatches[-1].platform_ref // "-"' 2>/dev/null || echo -)"
  node="$(printf '%s' "$detail" | jq -r '.dispatches[-1].nodes // "-"' 2>/dev/null || echo -)"
  echo "[forge-ci-e2e] poll $i/$POLL_MAX phase=$phase node=$node job=$jobref uid=$UID_"
  if [ "$phase" != "$prev_phase" ]; then
    echo "[forge-ci-e2e] phase ${prev_phase:-<start>} -> $phase"
    prev_phase="$phase"
  fi
  case "$phase" in
    Succeeded)
      summary "✅ **PASS** — forge-loop smoke completed. session_id=\`$UID_\` job=\`$jobref\`"
      post_status "success" "PASS — uid=${UID_}; job=${jobref}; sha=${HEAD_SHA:0:12}"
      forge_result="$(fetch_forge_result || true)"
      report_upsert "✅ Succeeded"
      exit 0 ;;
    Failed)
      err="$(printf '%s' "$detail" | jq -r '.orchestration.last_error // (.orchestration.conditions[-1].message) // "unknown"' 2>/dev/null)"
      summary "❌ **FAIL** — session_id=\`$UID_\` job=\`$jobref\` node=\`$node\`"
      summary "reason: $err"
      post_status "failure" "FAIL (${HEAD_SHA:0:12}): ${err:0:110}"
      report_upsert "❌ Failed"
      exit 1 ;;
  esac
  now_s="$(date +%s)"
  if [ $((now_s - last_push)) -ge "$STATUS_INTERVAL_S" ]; then
    post_status "pending" "running ${phase}; job=${jobref}; uid=${UID_}; sha=${HEAD_SHA:0:12}"
    last_push="$now_s"
  fi
  sleep "$POLL_INTERVAL_S"
done

summary "❌ **FAIL (timeout)** — workload did not reach terminal state. session_id=\`$UID_\`"
post_status "failure" "timeout; uid=${UID_}; sha=${HEAD_SHA:0:12}"
report_upsert "❌ Timeout"
if [ "${CI_E2E_DELETE_ON_TIMEOUT:-0}" = "1" ]; then
  cleanup
else
  summary "workload \`$UID_\` kept for triage"
fi
exit 1
