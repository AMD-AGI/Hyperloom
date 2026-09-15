#!/usr/bin/env bash
# End-to-end CI smoke test: submit ONE single-GPU inference-optimizer run built from
# *this PR's* branch, follow it to a verdict, and report it on the PR --
# Succeeded -> green / Failed|cancelled|timeout -> red.
#
# This script no longer talks to the run API. `dispatron-ci`, which the runner image
# carries, submits and polls; what is left here is the reporting the platform has no
# business owning: the live commit status, and one sticky comment per PR.
#
# The split is the point. The dispatch half was 295 lines of curl and jq against an
# endpoint owned by another team, duplicated per repository, and a fix to the polling
# or the error vocabulary had to be made once per copy. The reporting half is specific
# to how *this* repository wants its PRs annotated, and belongs here.
#
# What that costs: the phase names, the error vocabulary and the timings now arrive as
# `dispatron-ci`'s declared outputs rather than being read out of the API's JSON. That
# is a contract either way; the difference is that this one is versioned with the CLI
# and stated in `--help`, and the other one was a jq path into somebody else's
# response body.
#
# Requires: bash, curl, jq, and `dispatron-ci` on PATH (the runner image provides it).
#
# Inputs (env):
#   DISPATRON_BASE_URL  Dispatron's compat facade                     (required)
#   DISPATCH_TOKEN      bearer token, when the deployment wants one   (optional)
#   MODEL               HF repo id                       (default Qwen/Qwen3-0.6B)
#   MODEL_CLASS         dense|moe_mla|moe_swa|moe_mla_nsa|"" (default dense)
#   GPUS / TP           resources                        (default 1 / 1)
#   MAX_HOURS           optimizer time budget (hours)    (default 0.5)
#   PR_NUMBER           PR number (for the job name and the report comment)
#   HEAD_REF / HEAD_SHA the branch and the exact commit to run
#   HEAD_REPO_URL       PR head repo clone url (forks); token in userinfo is fine,
#                       the CLI lifts it out rather than putting it in `params`
#   BASE_REPO_URL       base repo clone url
#   MODEL_BASE          local model base dir (optional)
#   POLL_INTERVAL_S     seconds between polls             (default 30)
#   POLL_MAX            max polls before giving up        (default 120)
#   KNOWLEDGE_STORE_MODE  local|remote                    (default local)
#   KB_STORE_URL / KB_STORE_TOKEN  required together when mode=remote
#   CI_E2E_PR_CHECK_BASE  base dir for per-PR checkouts   (default /tmp/ci-e2e)
#   CI_E2E_INSECURE     skip TLS verification to the facade (default 0)
#
# Optional live commit status on the PR (all three required to enable):
#   GH_STATUS_TOKEN   GitHub token with statuses:write (Actions: secrets.GITHUB_TOKEN)
#   GH_STATUS_REPO    owner/repo
#   GH_STATUS_SHA     PR head sha to attach the status to
#   GH_STATUS_DETAILS_URL  link back to the Actions run (optional)
#   STATUS_CONTEXT    status context/name (default ci-e2e/run)
#   STATUS_INTERVAL_S seconds between status refreshes (default 300 => 5min)
# On terminal, a sticky PR report comment is upserted when GH_STATUS_TOKEN +
# GH_STATUS_REPO + numeric PR_NUMBER are set.
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen3-0.6B}"
MODEL_CLASS="${MODEL_CLASS:-dense}"
GPUS="${GPUS:-1}"
TP="${TP:-1}"
MAX_HOURS="${MAX_HOURS:-0.5}"
POLL_INTERVAL_S="${POLL_INTERVAL_S:-30}"
POLL_MAX="${POLL_MAX:-120}"
KNOWLEDGE_STORE_MODE="${KNOWLEDGE_STORE_MODE:-local}"

: "${DISPATRON_BASE_URL:?DISPATRON_BASE_URL is required}"
: "${HEAD_REF:?HEAD_REF (PR head branch) is required}"
: "${HEAD_SHA:?HEAD_SHA (immutable PR head commit) is required}"
case "$KNOWLEDGE_STORE_MODE" in
  local|remote) ;;
  *) echo "KNOWLEDGE_STORE_MODE must be local or remote" >&2; exit 2 ;;
esac
if [ "$KNOWLEDGE_STORE_MODE" = "remote" ]; then
  : "${KB_STORE_URL:?KB_STORE_URL is required when KNOWLEDGE_STORE_MODE=remote}"
  : "${KB_STORE_TOKEN:?KB_STORE_TOKEN is required when KNOWLEDGE_STORE_MODE=remote}"
fi

# Checked by name rather than left to fail as `command not found`, which reads like a
# broken workflow rather than a job that landed on a runner without the CLI.
command -v dispatron-ci >/dev/null 2>&1 || {
  echo "::error::dispatron-ci is not on PATH; this runner was not built from Dispatron's deploy/ci/runner.Dockerfile" >&2
  exit 2
}

# Fork PRs: the head branch lives in the contributor's fork, so clone from the head
# repo. Same-repo PRs use the base repo.
SRC_REPO="${BASE_REPO_URL:-}"
if [ -n "${HEAD_REPO_URL:-}" ] && [ "${HEAD_REPO_URL}" != "${BASE_REPO_URL:-}" ]; then
  SRC_REPO="${HEAD_REPO_URL}"
fi
# The exact commit is part of the source path. Do not reuse a branch-only checkout:
# a later push must never make this CI run test a different commit from the one its
# GitHub status is attached to.
PR_CHECK_BASE="${CI_E2E_PR_CHECK_BASE:-/tmp/ci-e2e}"
SRC_DIR="${CI_E2E_SOURCE_DIR:-${PR_CHECK_BASE%/}/pr_${PR_NUMBER:-manual}/${HEAD_SHA}/hyperloom}"

summary() { echo "$*" | tee -a "${GITHUB_STEP_SUMMARY:-/dev/null}"; }

# ---- GitHub commit status (optional live status on the PR) ----------------
# When GH_STATUS_TOKEN + GH_STATUS_REPO + GH_STATUS_SHA are set we publish a commit
# status against the PR head sha and refresh it every STATUS_INTERVAL_S, so the PR's
# checks section shows the live phase without opening the job log. No-op when absent.
STATUS_INTERVAL_S="${STATUS_INTERVAL_S:-300}"
STATUS_CONTEXT="${STATUS_CONTEXT:-ci-e2e/run}"
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
# (metadata + timeline). Needs GH_STATUS_TOKEN + GH_STATUS_REPO + numeric PR_NUMBER.
REPORT_MARKER="<!-- ci-e2e-report:${STATUS_CONTEXT} -->"
gh_report_on() { [ -n "${GH_STATUS_TOKEN:-}" ] && [ -n "${GH_STATUS_REPO:-}" ] && [[ "${PR_NUMBER:-}" =~ ^[0-9]+$ ]]; }
_epoch() { date -d "$1" +%s 2>/dev/null || true; }
_hdur() { local s="${1:-}"; [ -z "$s" ] && { echo "–"; return; }; if [ "$s" -lt 60 ]; then echo "${s}s"; else echo "$((s/60))m $((s%60))s"; fi; }

# `humanize_reason` used to live here, as a case statement over the platform's error
# strings. It is `dispatron-ci`'s `explanation` output now -- the same sentences, moved
# next to the vocabulary they describe, so the next team to wire up a GPU check gets
# them without copying this file.

report_upsert() { # result_md (e.g. "✅ Succeeded")
  gh_report_on || return 0
  local result="$1" qe de ee qd="" rt="" tot="" actions="" reason_row="" detail_block="" job_row="" body cid
  qe="$(_epoch "${OUT_queued_at:-}")"; de="$(_epoch "${OUT_dispatched_at:-}")"
  ee="$(_epoch "${OUT_ended_at:-}")"; [ -z "$ee" ] && ee="$(date +%s)"
  [ -n "$qe" ] && [ -n "$de" ] && qd="$(_hdur $((de - qe)))"
  [ -n "$de" ] && rt="$(_hdur $((ee - de)))"
  [ -n "$qe" ] && tot="$(_hdur $((ee - qe)))"
  [ -n "${GH_STATUS_DETAILS_URL:-}" ] && actions="[details](${GH_STATUS_DETAILS_URL})"
  if [ -n "${OUT_platform_ref:-}" ]; then
    job_row="| backend run | \`${OUT_platform_ref}\`${OUT_nodes:+ on \`${OUT_nodes}\`} |
"
  fi
  # A platform error is often a stack trace, and a newline inside a table cell ends the
  # row -- the old single-line `| detail | ... |` silently mangled the whole table the
  # first time one arrived. The sentence goes in the table; the raw text goes under it,
  # folded, where it can be as many lines as it likes.
  if [ -n "${OUT_explanation:-}" ]; then
    reason_row="| reason | $(printf '%s' "$OUT_explanation" | tr '\n' ' ') |
"
  fi
  if [ -n "${OUT_reason:-}" ]; then
    detail_block="
<details><summary>platform error</summary>

\`\`\`
$(printf '%s' "$OUT_reason" | head -c 4000)
\`\`\`

</details>
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
| commit | \`${HEAD_SHA}\` |
| session_id | \`${UID_:-–}\` |
${job_row}| queue → dispatch | ${qd:-–} |
| run time | ${rt:-–} |
| total | ${tot:-–} |
${reason_row}${detail_block}
${actions}"
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
# `dispatron-ci` submits and polls. Two channels come back out of it:
#   * the events file, one JSON object per poll, which is what the live status is
#     refreshed from while the run is still going;
#   * step outputs, which carry the verdict, the reason and the timings.
# Neither is this script parsing the CLI's prose, which is the coupling the split
# exists to remove.
WORK="$(mktemp -d)"
EVENTS="$WORK/events.jsonl"
# The CLI writes step outputs of its own. Pointed at a scratch file rather than the
# job's, so what this step exposes stays this script's contract and not a union of two.
OUTPUTS="$WORK/cli-outputs"
: > "$EVENTS"; : > "$OUTPUTS"
trap 'rm -rf "$WORK"' EXIT

insecure=()
[ "${CI_E2E_INSECURE:-0}" = "1" ] && insecure=(--insecure)
kb=()
if [ "$KNOWLEDGE_STORE_MODE" = "remote" ]; then
  kb=(--knowledge-mode remote --kb-url "$KB_STORE_URL" --kb-token "$KB_STORE_TOKEN")
fi

post_status "pending" "dispatching; sha=${HEAD_SHA:0:12}"

# The CLI's own step outputs must not land in the job's GITHUB_OUTPUT unread: this
# script reads them itself and re-exports the ones the workflow declares, so the file
# it writes is ours and the step's outputs stay this script's contract.
GITHUB_OUTPUT="$OUTPUTS" \
DISPATCH_EVENTS_FILE="$EVENTS" \
dispatron-ci \
  --base-url "$DISPATRON_BASE_URL" \
  --name "ci-e2e-pr${PR_NUMBER:-manual}-${HEAD_SHA:0:12}" \
  --source hyperloom-e2e \
  --model "$MODEL" \
  --model-class "$MODEL_CLASS" \
  --model-base "${MODEL_BASE:-}" \
  --gpus "$GPUS" \
  --tp "$TP" \
  --max-hours "$MAX_HOURS" \
  --user "${CI_E2E_USER_NAME:-}" \
  --ref "$HEAD_REF" \
  --sha "$HEAD_SHA" \
  --source-repo "$SRC_REPO" \
  --source-dir "$SRC_DIR" \
  --pr "${PR_NUMBER:-}" \
  --poll-interval "$POLL_INTERVAL_S" \
  --poll-max "$POLL_MAX" \
  "${kb[@]}" "${insecure[@]}" &
cli=$!

# Refresh the commit status while the run is going, from the events file rather than
# from a second poll of the API -- a status refresh must not be able to add load to
# the thing it is describing, nor to disagree with it.
last_push=0
while kill -0 "$cli" 2>/dev/null; do
  now_s="$(date +%s)"
  if [ $((now_s - last_push)) -ge "$STATUS_INTERVAL_S" ]; then
    ev="$(jq -c 'select(.event=="phase")' "$EVENTS" 2>/dev/null | tail -1 || true)"
    if [ -n "$ev" ]; then
      ph="$(printf '%s' "$ev" | jq -r '.phase // "?"' 2>/dev/null || echo '?')"
      jr="$(printf '%s' "$ev" | jq -r '.platform_ref // "" ' 2>/dev/null || true)"
      post_status "pending" "running ${ph}; job=${jr:--}; sha=${HEAD_SHA:0:12}"
    fi
    last_push="$now_s"
  fi
  sleep 5
done
rc=0; wait "$cli" || rc=$?

# Read what the CLI decided, from the terminal event rather than from the step-output
# file. Both carry the same fields, but the event is JSON: a platform error containing
# a newline needs no delimiter convention to survive, and reading one out of
# `key=value` means hand-rolling GitHub's escaping in bash to get it wrong once.
term="$(jq -c 'select(.event=="terminal")' "$EVENTS" 2>/dev/null | tail -1 || true)"
field() { [ -n "$term" ] && printf '%s' "$term" | jq -r --arg k "$1" '.[$k] // ""' 2>/dev/null || true; }

UID_="$(field uid)"
result="$(field result)"
OUT_reason="$(field reason)"
OUT_explanation="$(field explanation)"
OUT_platform_ref="$(field platform_ref)"
OUT_nodes="$(field nodes)"
OUT_queued_at="$(field queued_at)"
OUT_dispatched_at="$(field dispatched_at)"
OUT_ended_at="$(field ended_at)"
# The uid is known from the moment of submission, so prefer that over an outcome that
# may not exist: a failed dispatch still has to name what it tried, if anything.
[ -z "$UID_" ] && UID_="$(jq -r 'select(.event=="submitted")|.uid' "$EVENTS" 2>/dev/null | tail -1 || true)"
# The CLI died before writing an outcome -- a refused submit, or the facade unreachable.
# Not a run that failed, and saying so stops somebody debugging code that never ran.
[ -z "$result" ] && result="dispatch-error"

# Re-export the outputs the workflow's later steps read.
if [ -n "${GITHUB_OUTPUT:-}" ] && [ "${GITHUB_OUTPUT}" != "$OUTPUTS" ]; then
  {
    echo "session_id=${UID_}"
    echo "result=${result}"
    echo "platform_ref=${OUT_platform_ref:-}"
  } >> "$GITHUB_OUTPUT"
fi

case "$result" in
  succeeded)
    summary "✅ **PASS** — run completed. session_id=\`${UID_}\` job=\`${OUT_platform_ref:--}\`"
    post_status "success" "PASS — uid=${UID_}; job=${OUT_platform_ref:--}; sha=${HEAD_SHA:0:12}"
    report_upsert "✅ Succeeded" ;;
  cancelled)
    # Not a red build: a newer commit or a `/retest` cancelled this one, and reporting
    # it as a failure sends somebody looking for a bug that is not there.
    summary "🚫 **CANCELLED** — session_id=\`${UID_}\`"
    post_status "error" "cancelled; uid=${UID_}; sha=${HEAD_SHA:0:12}"
    report_upsert "🚫 Cancelled" ;;
  timeout)
    summary "❌ **FAIL (timeout)** — ${OUT_explanation:-gave up waiting}. session_id=\`${UID_}\`"
    post_status "failure" "timeout; uid=${UID_}; sha=${HEAD_SHA:0:12}"
    report_upsert "⏱ Timed out" ;;
  dispatch-error)
    summary "❌ **FAIL** — the run was never dispatched; see the job log."
    post_status "error" "could not dispatch; sha=${HEAD_SHA:0:12}"
    report_upsert "❌ Not dispatched" ;;
  *)
    summary "❌ **FAIL** — session_id=\`${UID_}\` job=\`${OUT_platform_ref:--}\` node=\`${OUT_nodes:--}\`"
    summary "reason: ${OUT_explanation:-unknown}"
    post_status "failure" "FAIL (${HEAD_SHA:0:12}): ${OUT_explanation:-unknown}"
    report_upsert "❌ Failed" ;;
esac

exit "$rc"
