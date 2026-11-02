#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# Publish the review-pr card produced by pr-review.yml.
#
# The verdict is applied only when the card passes the skill's check-card.sh, held to a risk graded here from the PR's
# files on GitHub rather than from anything the review job wrote; names the clean skill version and the model this
# run recorded in skill.txt; and still describes the open PR head:
#   Blocking issues: N     -> request-changes review (blocks the merge)
#   Blocking issues: none  -> dismiss this bot's CHANGES_REQUESTED reviews, then an LGTM comment
# Anything else posts a note (with the card, if any) and exits 1. The bot never approves, whatever the risk.
#
# env: REPO, PR, REVIEWED_SHA, CARD, RUN_URL, REVIEW_RESULT (result of the review job)

set -euo pipefail

BOT_LOGIN='github-actions[bot]'
SKILL_DIR="$(dirname "${BASH_SOURCE[0]}")/../../.claude/skills/review-pr"

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

WORK=$(dirname "$CARD")
[ -s "$WORK/skill.txt" ] || note "the review left no skill.txt"
skill=$(sed -n 's/^skill: //p' "$WORK/skill.txt")
model=$(sed -n 's/^model: //p' "$WORK/skill.txt")
[[ "$skill" =~ ^[0-9a-f]{12,40}$ ]] || note "skill.txt names no clean skill version: '${skill}'"
if [ -z "$model" ] || [ "$model" = unknown ]; then
  note "skill.txt names no model"
fi

graded=$(mktemp -d)
trap 'rm -rf "$graded"' EXIT
gh api --paginate "repos/$REPO/pulls/$PR/files" --jq '.[] | .filename, (.previous_filename // empty)' \
  > "$graded/files.txt" \
  || note "listing the PR's files failed"
python3 "$SKILL_DIR/risk.py" "$graded/files.txt" > "$graded/risk.txt" || note "risk.py could not grade the PR's files"
defect=$(bash "$SKILL_DIR/check-card.sh" "$WORK" "$PR" "$REVIEWED_SHA" "$graded/risk.txt") || note "$defect"
grep -qF " | Skill: ${skill} | Model: ${model} | " "$CARD" \
  || note "the card's Checked line does not name this run's Skill: ${skill} and Model: ${model}"

pr_now=$(gh pr view "$PR" --repo "$REPO" --json state,headRefOid --jq '"\(.state) \(.headRefOid)"') \
  || note "reading the PR state failed"
read -r state head <<< "$pr_now"
[ "$state" = OPEN ] || note "the PR is ${state}"
[ "$head" = "$REVIEWED_SHA" ] || note "the PR head moved from ${REVIEWED_SHA} to ${head} during the review"

if ! grep -q '^Blocking issues: none$' "$CARD"; then
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
