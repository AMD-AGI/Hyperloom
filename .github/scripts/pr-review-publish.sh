#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# Publish the review-pr card produced by pr-review.yml.
#
# The verdict is applied only when the card parses strictly and still describes the open PR head:
#   Blocking issues: N     -> request-changes review (blocks the merge)
#   Blocking issues: none  -> dismiss this bot's CHANGES_REQUESTED reviews, then an LGTM comment
# Anything else posts a note (with the card, if any) and exits 1. The bot never approves.
#
# env: REPO, PR, REVIEWED_SHA, CARD, RUN_URL, REVIEW_RESULT (result of the review job)

set -euo pipefail

BOT_LOGIN='github-actions[bot]'

note() {
  local body="Automated review: no verdict applied -- $1. Comment \`/review\` to retry. Run: ${RUN_URL}"
  if [ -s "$CARD" ]; then
    body="${body}"$'\n\n'"$(cat "$CARD")"
  fi
  gh pr comment "$PR" --repo "$REPO" --body "$body" || echo "pr-review-publish: posting the note failed" >&2
  echo "pr-review-publish: $1" >&2
  exit 1
}

[ "$REVIEW_RESULT" = success ] || note "the review job ended with result '${REVIEW_RESULT}'"
[ -s "$CARD" ] || note "the review produced no card"

WORK=$(dirname "$CARD")
required=(rules.txt answers.txt core_files.txt verdicts.txt ai_diagnostic.txt refutations.txt independent.txt)
for artifact in "${required[@]}"; do
  [ -s "$WORK/$artifact" ] || note "the review did not produce non-empty ${artifact}"
done

first=$(head -1 "$CARD")
[[ "$first" =~ ^##\ PR\ \#$PR\ --\ .+ ]] || note "the card has no well-formed PR heading"
[ "$(grep -cE '^What it does: .+' "$CARD" || true)" = 1 ] \
  || note "the card has no single well-formed 'What it does:' line"
strict=$(grep -cE '^Blocking issues: (none|[1-5])$' "$CARD" || true)
loose=$(grep -ciE '^[#*[:space:]]*blocking issues' "$CARD" || true)
[ "$strict" = 1 ] && [ "$loose" = 1 ] \
  || note "the card has no single well-formed 'Blocking issues:' line"

verdict=$(sed -nE 's/^Blocking issues: (none|[1-5])$/\1/p' "$CARD")
findings=$(grep -cE '^[0-9]+\. \[' "$CARD" || true)
if [ "$verdict" = none ]; then expected=0; else expected=$verdict; fi
[ "$findings" = "$expected" ] \
  || note "the card states 'Blocking issues: ${verdict}' but lists ${findings} numbered finding(s)"
verified=$(grep -cE '^[0-9]+\. \[[^]]+\] .+ \[verified\]$' "$CARD" || true)
[ "$verified" = "$expected" ] || note "every blocking finding must be marked [verified]"
for field in Problem Impact Action; do
  [ "$(grep -cE "^   ${field}: .+" "$CARD" || true)" = "$expected" ] \
    || note "every blocking finding must carry ${field}"
done
if [ "$verdict" = none ]; then
  ! grep -q '^deferred:' "$CARD" || note "a clean card cannot defer a surviving finding"
fi
grep -qE "^Checked: .+ \\| Ran: .+ \\| Base: [0-9a-f]{7,40} \\| Head: ${REVIEWED_SHA}$" "$CARD" \
  || note "the card has no well-formed verification-depth line for the reviewed head"

pr_now=$(gh pr view "$PR" --repo "$REPO" --json state,headRefOid --jq '"\(.state) \(.headRefOid)"') \
  || note "reading the PR state failed"
read -r state head <<< "$pr_now"
[ "$state" = OPEN ] || note "the PR is ${state}"
[ "$head" = "$REVIEWED_SHA" ] || note "the PR head moved from ${REVIEWED_SHA} to ${head} during the review"

if [ "$verdict" != none ]; then
  gh pr review "$PR" --repo "$REPO" --request-changes --body-file "$CARD" \
    || note "submitting the request-changes review failed"
  exit 0
fi

ids=$(gh api --paginate "repos/$REPO/pulls/$PR/reviews" \
  --jq ".[] | select(.user.login == \"$BOT_LOGIN\" and .state == \"CHANGES_REQUESTED\") | .id") \
  || note "listing earlier reviews failed; LGTM withheld"
for id in $ids; do
  gh api -X PUT "repos/$REPO/pulls/$PR/reviews/$id/dismissals" \
    -f message="Superseded by a clean automated review at ${REVIEWED_SHA}." -f event=DISMISS >/dev/null \
    || note "dismissing earlier review ${id} failed; LGTM withheld"
done
gh pr comment "$PR" --repo "$REPO" --body "$(cat "$CARD")"$'\n\nLGTM' \
  || note "posting the LGTM comment failed"
