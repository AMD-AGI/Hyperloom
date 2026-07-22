#!/usr/bin/env bash
# End-to-end CI smoke test: submit ONE single-GPU Hyperloom inference-optimizer
# run built from *this PR's* branch to the robust-spur orchestration API, poll
# until terminal, and map Succeeded->green / Failed|timeout->red. The run's
# workload uid == CLAW_SESSION_ID (the spur task id), printed on every path.
#
# Requires: bash, curl, jq on the (self-hosted, in-network) runner.
#
# Inputs (env):
#   ROBUST_BASE       robust-spur ingress base (default crusoe ingress)
#   SAFE_AK           SaFE API key (ak-...), required
#   MODEL             HF repo id                     (default Qwen/Qwen3-0.6B)
#   MODEL_CLASS       dense|moe_mla|moe_swa|moe_mla_nsa|"" (default dense; "" -> auto-infer)
#   GPUS              physical GPUs per replica       (default 1)
#   TP                tensor-parallel degree          (default 1)
#   MAX_HOURS         optimizer time budget (hours)   (default 0.5)
#   PR_NUMBER         PR number (for the job name)
#   HEAD_REF          PR head branch name (the code to run)
#   HEAD_REPO_URL     PR head repo clone url (for forks)
#   BASE_REPO_URL     base repo clone url (AMD-AGI/Hyperloom)
#   POLL_INTERVAL_S   seconds between polls            (default 30)
#   POLL_MAX          max polls before timeout         (default 120 => ~60min)
#   CI_E2E_PR_CHECK_BASE  shared base for per-PR checkouts (default /shared_nfs/pr_check)
#
# Optional live commit status on the PR (all three required to enable):
#   GH_STATUS_TOKEN   GitHub token with statuses:write (Actions: secrets.GITHUB_TOKEN)
#   GH_STATUS_REPO    owner/repo (e.g. AMD-AGI/Hyperloom)
#   GH_STATUS_SHA     PR head sha to attach the status to
#   GH_STATUS_DETAILS_URL  link back to the Actions run (optional)
#   STATUS_CONTEXT    status context/name (default ci-e2e/hyperloom-run)
#   STATUS_INTERVAL_S seconds between status refreshes (default 300 => 5min)
# On terminal, a sticky PR report comment is upserted when GH_STATUS_TOKEN +
# GH_STATUS_REPO + numeric PR_NUMBER are set (optional PULSE_SESSION_BASE to link out).
set -euo pipefail

ROBUST_BASE="${ROBUST_BASE:-https://crusoe.primus-safe.amd.com/robust-spur-api}"
MODEL="${MODEL:-Qwen/Qwen3-0.6B}"
MODEL_CLASS="${MODEL_CLASS:-dense}"
GPUS="${GPUS:-1}"
TP="${TP:-1}"
MAX_HOURS="${MAX_HOURS:-0.5}"
POLL_INTERVAL_S="${POLL_INTERVAL_S:-30}"
POLL_MAX="${POLL_MAX:-120}"
API="${ROBUST_BASE%/}/api/v1/orchestration/workloads"

: "${SAFE_AK:?SAFE_AK (SaFE API key) is required}"
: "${HEAD_REF:?HEAD_REF (PR head branch) is required}"

# Fork PRs: the head branch lives in the contributor's fork, not AMD-AGI/Hyperloom,
# so clone from the head repo. Same-repo PRs use the base repo.
SRC_REPO="${BASE_REPO_URL:-https://github.com/AMD-AGI/Hyperloom.git}"
if [ -n "${HEAD_REPO_URL:-}" ] && [ "${HEAD_REPO_URL}" != "${BASE_REPO_URL:-}" ]; then
  SRC_REPO="${HEAD_REPO_URL}"
fi
# Where the in-container launcher puts this PR's Hyperloom source. We give every PR
# a stable, inspectable path under a shared base so runs are easy to find/re-clone:
#     <base>/pr_<N>/hyperloom   (e.g. /shared_nfs/pr_check/pr_971/hyperloom)
# Override the base with CI_E2E_PR_CHECK_BASE. Point CI_E2E_SOURCE_DIR at an existing
# checkout to skip cloning entirely. Re-triggered runs pick up new commits because the
# launcher refreshes this checkout when HL_CI_E2E=1 (see _incontainer.sh.in).
PR_CHECK_BASE="${CI_E2E_PR_CHECK_BASE:-/shared_nfs/pr_check}"
SRC_DIR="${CI_E2E_SOURCE_DIR:-${PR_CHECK_BASE%/}/pr_${PR_NUMBER:-manual}/hyperloom}"

summary() { echo "$*" | tee -a "${GITHUB_STEP_SUMMARY:-/dev/null}"; }
auth=(-H "Authorization: Bearer ${SAFE_AK}")

# TLS to the internal ingress: prefer a CA bundle (CI_E2E_CACERT), fall back to
# skip-verify only when CI_E2E_INSECURE=1 (test convenience on trusted networks).
tls=()
if [ -n "${CI_E2E_CACERT:-}" ]; then
  tls=(--cacert "$CI_E2E_CACERT")
elif [ "${CI_E2E_INSECURE:-0}" = "1" ]; then
  tls=(-k)
fi

# ---- GitHub commit status (optional live status on the PR) ----------------
# When GH_STATUS_TOKEN + GH_STATUS_REPO + GH_STATUS_SHA are set we publish a commit
# status against the PR head sha and refresh it every STATUS_INTERVAL_S, so the PR's
# checks section shows the live phase without opening the job log. Commit statuses work
# with both the Actions GITHUB_TOKEN and a personal token. No-op when vars are absent.
STATUS_INTERVAL_S="${STATUS_INTERVAL_S:-300}"
STATUS_CONTEXT="${STATUS_CONTEXT:-ci-e2e/hyperloom-run}"
GH_API="${GH_API:-https://api.github.com}"
gh_status_on() { [ -n "${GH_STATUS_TOKEN:-}" ] && [ -n "${GH_STATUS_REPO:-}" ] && [ -n "${GH_STATUS_SHA:-}" ]; }
post_status() { # state(pending|success|failure|error)  description
  gh_status_on || return 0
  local desc="${2:0:139}"
  curl -sS -X POST \
    -H "Authorization: Bearer ${GH_STATUS_TOKEN}" \
    -H "Accept: application/vnd.github+json" \
    -H "X-GitHub-Api-Version: 2022-11-28" \
    "${GH_API}/repos/${GH_STATUS_REPO}/statuses/${GH_STATUS_SHA}" \
    -d "$(jq -n --arg s "$1" --arg d "$desc" --arg u "${GH_STATUS_DETAILS_URL:-}" --arg c "$STATUS_CONTEXT" \
        '{state:$s, description:$d, context:$c} + (if $u=="" then {} else {target_url:$u} end)')" \
    >/dev/null 2>&1 || true
}

# ---- GitHub PR report comment (optional) ----------------------------------
# On terminal, upsert ONE sticky comment on the PR with a compact run report
# (metadata + timeline derived from the spur API). Needs GH_STATUS_TOKEN +
# GH_STATUS_REPO + a numeric PR_NUMBER. Optional: PULSE_SESSION_BASE to link out
# to the full Pulse dashboard by claw_session_id. No-op when prerequisites absent.
REPORT_MARKER="<!-- ci-e2e-report:${STATUS_CONTEXT} -->"
gh_report_on() { [ -n "${GH_STATUS_TOKEN:-}" ] && [ -n "${GH_STATUS_REPO:-}" ] && [[ "${PR_NUMBER:-}" =~ ^[0-9]+$ ]]; }
_epoch() { date -d "$1" +%s 2>/dev/null || true; }
_hdur() { local s="${1:-}"; [ -z "$s" ] && { echo "–"; return; }; if [ "$s" -lt 60 ]; then echo "${s}s"; else echo "$((s/60))m $((s%60))s"; fi; }

# Turn a raw platform error into a one-line, plain-language reason a reviewer can act on.
humanize_reason() {
  case "$1" in
    *JobHoldMaxRequeue*)
      echo "Slurm kept holding & requeuing the job until it gave up — usually a flaky node; just re-run." ;;
    *baseline_accuracy*|*accuracy_failed*)
      echo "Baseline accuracy gate failed — the model server ran but the baseline benchmark didn't pass; check the baseline logs." ;;
    *NonZeroExitCode*|*"exhausted retries"*)
      echo "The run crashed on the compute node (non-zero exit) — usually a node/env hiccup (e.g. a leftover process holding a port) or a runtime error; often transient, re-run first." ;;
    *"not terminal"*|*[Tt]imeout*)
      echo "Timed out — the run never reached a terminal state in time (task stuck, or the GPU stayed queued too long)." ;;
    *"not associated"*|*user_name*)
      echo "Dispatch identity rejected — CI_E2E_USER_NAME must be a member of the amd-hyperloom Slurm account." ;;
    *128*|*Authentication*|*"could not read Username"*)
      echo "Could not clone the PR code on the compute node — GitHub auth/permission issue." ;;
    "")
      echo "Unknown failure — the platform reported no error detail." ;;
    *)
      echo "$1" ;;
  esac
}

report_upsert() { # result_md (e.g. "✅ Succeeded")
  gh_report_on || return 0
  local result="$1" q d qe de nows qd="" rt="" tot="" pulse="" actions="" reason_row="" body cid
  q="$(printf '%s' "${detail:-}" | jq -r '[.orchestration.conditions[]|select(.phase=="Queued")][0].time // empty' 2>/dev/null || true)"
  d="$(printf '%s' "${detail:-}" | jq -r '[.orchestration.conditions[]|select(.phase=="Dispatched")][0].time // empty' 2>/dev/null || true)"
  qe="$(_epoch "$q")"; de="$(_epoch "$d")"; nows="$(date +%s)"
  [ -n "$qe" ] && [ -n "$de" ] && qd="$(_hdur $((de - qe)))"
  [ -n "$de" ] && rt="$(_hdur $((nows - de)))"
  [ -n "$qe" ] && tot="$(_hdur $((nows - qe)))"
  [ -n "${PULSE_SESSION_BASE:-}" ] && pulse="· [Pulse session](${PULSE_SESSION_BASE%/}/${UID_})"
  [ -n "${GH_STATUS_DETAILS_URL:-}" ] && actions="[details](${GH_STATUS_DETAILS_URL})"
  if [ -n "${err:-}" ]; then
    reason_row="| reason | $(humanize_reason "$err") |
| detail | \`${err}\` |
"
  fi
  body="${REPORT_MARKER}
## CI E2E report — ${result}

| item | value |
|---|---|
| result | ${result} |
| model | \`${MODEL}\` (${MODEL_CLASS:-dense}) |
| resources | ${GPUS}× GPU, TP=${TP} |
| PR branch | \`${HEAD_REF}\` |
| claw_session_id | \`${UID_}\` |
| queue → dispatch | ${qd:-–} |
| run time | ${rt:-–} |
| total | ${tot:-–} |
${reason_row}
${actions} ${pulse}"
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

# ---- submit ---------------------------------------------------------------
params="$(jq -n \
  --arg repo_id "$MODEL" --arg tp "$TP" --arg mh "$MAX_HOURS" \
  --arg ref "$HEAD_REF" --arg srcrepo "$SRC_REPO" --arg srcdir "$SRC_DIR" \
  --arg mc "$MODEL_CLASS" --arg mbase "${CI_E2E_MODEL_BASE:-}" \
  '{REPO_ID:$repo_id, TP:$tp, MAX_HOURS:$mh,
    HYPERLOOM_SOURCE_REF:$ref, HYPERLOOM_SOURCE_REPO:$srcrepo, HYPERLOOM_SOURCE_DIR:$srcdir}
   + (if $mc == "" then {} else {MODEL_CLASS:$mc} end)
   + (if $mbase == "" then {} else {HL_MODEL_BASE:$mbase} end)')"
body="$(jq -n \
  --arg name "ci-pr-${PR_NUMBER:-manual}-${GITHUB_RUN_ID:-local}" \
  --arg uname "${CI_E2E_USER_NAME:-}" \
  --argjson gpus "$GPUS" --argjson params "$params" \
  '{name:$name, infra_type:"spur", kind:"hyperloom", replicas:1,
    gpu_per_replica:$gpus, template:{params:$params, env:{HL_CI_E2E:"1"}}}
   + (if $uname == "" then {} else {user_name:$uname} end)')"

echo "[ci-e2e] submitting: model=$MODEL ref=$HEAD_REF gpus=$GPUS tp=$TP max_hours=$MAX_HOURS"
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
summary "**claw_session_id (workload uid):** \`$UID_\`"
echo "claw_session_id=$UID_" >> "${GITHUB_OUTPUT:-/dev/null}"
# Persist the uid so a workflow `if: cancelled()` step can DELETE the workload even
# if this process is hard-killed on cancel (the trap below is best-effort only).
echo "$UID_" > "${CLAW_UID_FILE:-${RUNNER_TEMP:-/tmp}/claw_uid}" 2>/dev/null || true

# Seed the live commit status (no-op unless GH_STATUS_* are set).
post_status "pending" "submitted; uid=${UID_}; model=${MODEL} ref=${HEAD_REF}"
last_push="$(date +%s)"

# On cancellation (e.g. a newer commit via concurrency cancel-in-progress),
# best-effort cancel the spur workload so we don't leak a GPU run.
cleanup() { curl -sS "${tls[@]}" -X DELETE "$API/$UID_" "${auth[@]}" >/dev/null 2>&1 || true; }
trap 'echo "[ci-e2e] cancelled; deleting workload $UID_"; post_status "error" "cancelled (newer commit or manual stop); uid=${UID_}"; cleanup; exit 1' INT TERM

# ---- poll -----------------------------------------------------------------
i=0
prev_phase=""
while [ "$i" -lt "$POLL_MAX" ]; do
  i=$((i + 1))
  detail="$(curl -sS "${tls[@]}" "$API/$UID_" "${auth[@]}" || true)"
  phase="$(printf '%s' "$detail" | jq -r '.orchestration.phase // "Unknown"' 2>/dev/null || echo Unknown)"
  spur="$(printf '%s' "$detail" | jq -r '.dispatches[-1].platform_ref // "-"' 2>/dev/null || echo -)"
  node="$(printf '%s' "$detail" | jq -r '.dispatches[-1].nodes // "-"' 2>/dev/null || echo -)"
  echo "[ci-e2e] poll $i/$POLL_MAX phase=$phase node=$node spur_job=$spur uid=$UID_"
  # Announce phase transitions (from the orchestration conditions ledger).
  if [ "$phase" != "$prev_phase" ]; then
    cond="$(printf '%s' "$detail" | jq -r '.orchestration.conditions[-1] | "\(.time) \(.phase): \(.message)"' 2>/dev/null || echo "")"
    echo "[ci-e2e]   >> phase ${prev_phase:-<start>} -> ${phase}${cond:+   ($cond)}"
    prev_phase="$phase"
  fi
  case "$phase" in
    Succeeded)
      summary "✅ **PASS** — run completed. claw_session_id=\`$UID_\` spur_job=\`$spur\`"
      post_status "success" "PASS — Succeeded; uid=${UID_}; spur_job=${spur}"
      report_upsert "✅ Succeeded"
      exit 0 ;;
    Failed)
      err="$(printf '%s' "$detail" | jq -r '.orchestration.last_error // (.orchestration.conditions[-1].message) // "unknown"' 2>/dev/null)"
      summary "❌ **FAIL** — claw_session_id=\`$UID_\` spur_job=\`$spur\` node=\`$node\`"
      summary "reason: $err"
      post_status "failure" "$(humanize_reason "$err")"
      report_upsert "❌ Failed"
      exit 1 ;;
  esac
  # Throttled live status: push at most once per STATUS_INTERVAL_S (default 5min).
  now_s="$(date +%s)"
  if [ $((now_s - last_push)) -ge "$STATUS_INTERVAL_S" ]; then
    post_status "pending" "running phase=${phase} (poll ${i}/${POLL_MAX}); spur_job=${spur}; uid=${UID_}"
    last_push="$now_s"
  fi
  sleep "$POLL_INTERVAL_S"
done

err="not terminal after $((POLL_MAX * POLL_INTERVAL_S))s"
summary "❌ **FAIL (timeout)** — ${err}. claw_session_id=\`$UID_\`"
post_status "failure" "FAIL (timeout) after $((POLL_MAX * POLL_INTERVAL_S))s; uid=${UID_}"
report_upsert "❌ Timeout"
cleanup
exit 1
