#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# Check a review-pr card against the skill's card contract before it is published. CI's
# pr-review-publish.sh and a local run both call this; it prints the first defect and exits 1.
#
# usage: check-card.sh <WORK> <PR> <HEAD> <RISK_TXT>
#   WORK      the review's scratch directory: card.md beside the artifacts the skill writes
#   RISK_TXT  the risk.txt the card's Risk may not fall below; CI grades one from the PR's files on GitHub

set -euo pipefail

fail() {
  echo "$1"
  exit 1
}

[ "$#" = 4 ] || fail "usage: check-card.sh <WORK> <PR> <HEAD> <RISK_TXT>"
WORK=$1
PR=$2
HEAD=$3
RISK=$4
CARD="$WORK/card.md"

[ -s "$CARD" ] || fail "the review produced no card"
required=(rules.txt answers.txt core_files.txt verdicts.txt ai_diagnostic.txt refutations.txt independent.txt)
for artifact in "${required[@]}"; do
  [ -s "$WORK/$artifact" ] || fail "the review did not produce non-empty ${artifact}"
done

# Any heading for this PR counts as an earlier card (references/re-review.md): earlier cards include ones people
# posted from local runs, and a match stricter than theirs would turn this check off unseen.
[ -f "$WORK/comments.txt" ] || fail "the review left no comments.txt"
if grep -qE "^## PR #${PR}([^0-9]|$)" "$WORK/comments.txt"; then
  verdicts="$WORK/prior_verdicts.txt"
  [ -s "$verdicts" ] || fail "the PR has an earlier card but the review produced no prior_verdicts.txt"
  if grep -qvE '^([^ ]+ (FIXED|HOLDS|MOVED)|none) -- .+$' "$verdicts" \
    || { grep -q '^none -- ' "$verdicts" && [ "$(grep -c '' "$verdicts")" -gt 1 ]; }; then
    fail "prior_verdicts.txt is neither one '<KEY> FIXED|HOLDS|MOVED -- <evidence>' line per finding nor one 'none --' line"
  fi
  prior=$(sed -E -e 's/^([^ ]+) FIXED -- .*/\1 fixed/' -e 's/^([^ ]+) HOLDS -- .*/\1 holds/' \
    -e 's/^([^ ]+) MOVED -- .*/\1 moved/' -e 's/^none -- .*/none/' "$verdicts" | paste -sd , - | sed 's/,/, /g')
  grep -qF " | Prior: ${prior} | Base: " "$CARD" \
    || fail "the card's Prior does not list prior_verdicts.txt: expected 'Prior: ${prior}'"
  # A HOLDS or MOVED finding must be back on the card; otherwise a clean card would dismiss the bot's earlier
  # request-changes review while Prior itself says the defect is still there.
  while read -r key; do
    grep -qE "^[0-9]+\\. \\[${key}\\] " "$CARD" \
      || fail "prior_verdicts.txt says ${key} still holds but the card does not list it"
  done < <(sed -nE 's/^([^ ]+) (HOLDS|MOVED) -- .*/\1/p' "$verdicts")
elif grep -qF ' | Prior: ' "$CARD"; then
  fail "the card names a Prior but the PR has no earlier card"
fi

first=$(head -1 "$CARD")
[[ "$first" =~ ^##\ PR\ \#$PR\ --\ .+ ]] || fail "the card has no well-formed PR heading"
[ "$(grep -cE '^What it does: .+' "$CARD" || true)" = 1 ] \
  || fail "the card has no single well-formed 'What it does:' line"

# The card's Risk is risk.txt's level, or higher with the level it raised from; never lower.
floor=$(sed -n 's/^risk: //p' "$RISK")
[[ "$floor" =~ ^(low|medium|high)$ ]] || fail "${RISK} names no risk"
[ "$(grep -c '^Risk: ' "$CARD" || true)" = 1 ] || fail "the card has no single 'Risk:' line"
[[ "$(grep '^Risk: ' "$CARD")" =~ ^Risk:\ (low|medium|high)(\ \(raised\ from\ (low|medium)\))?\ --\ .+$ ]] \
  || fail "the card's Risk line is not 'Risk: <low|medium|high>[ (raised from <level>)] -- <reason>'"
risk=${BASH_REMATCH[1]}
raised=${BASH_REMATCH[3]}
rank() { case "$1" in low) echo 0 ;; medium) echo 1 ;; high) echo 2 ;; esac }
if [ "$risk" = "$floor" ]; then
  [ -z "$raised" ] || fail "the card raises its Risk from ${raised}, but the changed paths grade it ${floor}"
elif [ "$(rank "$risk")" -lt "$(rank "$floor")" ]; then
  fail "the card's Risk ${risk} is below ${floor}, the risk its changed paths carry"
else
  [ "$raised" = "$floor" ] || fail "the card's Risk ${risk} is above ${floor} and must say '(raised from ${floor})'"
fi

headers=$(grep -c '^Decisions for a human:' "$CARD" || true)
if [ "$risk" = low ]; then
  [ "$headers" = 0 ] || fail "a low-risk card has no 'Decisions for a human:'; a decision a person must make raises it"
elif grep -qE '^Decisions for a human: none -- .+$' "$CARD"; then
  [ "$headers" = 1 ] || fail "the card has more than one 'Decisions for a human:' list"
  [ "$risk" = medium ] || fail "a high-risk card names at least one decision for a human"
else
  if [ "$headers" != 1 ] || ! grep -qx 'Decisions for a human:' "$CARD"; then
    fail "a ${risk}-risk card has one 'Decisions for a human:' list, or 'Decisions for a human: none -- <why>'"
  fi
  decisions=$(awk '/^Decisions for a human:$/ { on = 1; next } on && /^- / { print; next } on { exit }' "$CARD")
  [ -n "$decisions" ] || fail "the card's 'Decisions for a human:' list is empty"
  ! grep -qvE '^- [^ ].* -- .+ -- .+$' <<< "$decisions" \
    || fail "every decision is '- <paths> -- <what changes> -- <the question a person answers>'"
  # Every high path risk.txt names is covered: named whole, or under a directory named with its trailing slash.
  named=$(sed -E 's/^- //; s/ -- .*//; s/`//g' <<< "$decisions" | tr ',' '\n' | sed -E 's/^ +//; s/ +$//')
  while read -r path; do
    covered=false
    while read -r name; do
      if [ "$path" = "$name" ] || { [[ "$name" == */ ]] && [[ "$path" == "$name"* ]]; }; then
        covered=true
        break
      fi
    done <<< "$named"
    $covered || fail "high-risk path ${path} is named by no decision for a human"
  done < <(sed -nE 's/^high ([^ ]+) -- .*/\1/p' "$RISK")
fi

strict=$(grep -cE '^Blocking issues: (none|[1-5])$' "$CARD" || true)
loose=$(grep -ciE '^[#*[:space:]]*blocking issues' "$CARD" || true)
[ "$strict" = 1 ] && [ "$loose" = 1 ] \
  || fail "the card has no single well-formed 'Blocking issues:' line"

verdict=$(sed -nE 's/^Blocking issues: (none|[1-5])$/\1/p' "$CARD")
findings=$(grep -cE '^[0-9]+\. \[' "$CARD" || true)
if [ "$verdict" = none ]; then expected=0; else expected=$verdict; fi
[ "$findings" = "$expected" ] \
  || fail "the card states 'Blocking issues: ${verdict}' but lists ${findings} numbered finding(s)"
verified=$(grep -cE '^[0-9]+\. \[[^]]+\] .+ \[verified\]$' "$CARD" || true)
[ "$verified" = "$expected" ] || fail "every blocking finding must be marked [verified]"
for field in Problem Impact Action; do
  [ "$(grep -cE "^   ${field}: .+" "$CARD" || true)" = "$expected" ] \
    || fail "every blocking finding must carry ${field}"
done
if [ "$verdict" = none ]; then
  ! grep -q '^deferred:' "$CARD" || fail "a clean card cannot defer a surviving finding"
fi
if grep -q '^Trim in this PR:$' "$CARD"; then
  [ "$(grep -c '^Trim in this PR:$' "$CARD")" = 1 ] || fail "the card has more than one 'Trim in this PR:' list"
  trims=$(awk '/^Trim in this PR:$/ { on = 1; next } on && /^- / { print; next } on { exit }' "$CARD")
  count=$(grep -c . <<< "$trims" || true)
  [ "$count" -ge 1 ] && [ "$count" -le 5 ] \
    || fail "the card's 'Trim in this PR:' list holds ${count} line(s); it holds 1 to 5, or is left out"
  ! grep -qvE '^- [^ ]+:[0-9]+(-[0-9]+)? -- .+ -- .+$' <<< "$trims" \
    || fail "every 'Trim in this PR:' line is '- <path>:<line>[-<line>] -- <what to delete> -- <evidence>'"
fi

grep -qE "^Checked: .+ \\| Ran: .+ \\| Base: [0-9a-f]{7,40} \\| Head: ${HEAD}$" "$CARD" \
  || fail "the card has no well-formed verification-depth line for the reviewed head"
[ -s "$WORK/skill.txt" ] || fail "the review left no skill.txt"
skill=$(sed -n 's/^skill: //p' "$WORK/skill.txt")
grep -qF " | Skill: ${skill} | Model: " "$CARD" \
  || fail "the card's Checked line does not name this run's Skill: ${skill}"
model=$(sed -nE 's/^Checked: .* \| Model: ([^|]*) \| .*/\1/p' "$CARD")
[ -n "${model// /}" ] && [ "$model" != unknown ] || fail "the card's Checked line names no model"
