#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# Step 4 of the check-prior-art skill: collect one candidate pull request for a deep read.
#
# usage: read.sh WORK OWNER/NAME NUMBER
# Writes WORK/prs/OWNER/NAME/NUMBER/ as Step 4 of SKILL.md describes, and prints that directory
# last. A diff that cannot be read becomes a SKIPPED line in diff.txt; any other part that cannot
# be collected exits non-zero.

set -euo pipefail

# shellcheck disable=SC1091  # shellcheck checks lib.sh as a file of its own
. "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

[ "$#" -eq 3 ] || die "usage: read.sh WORK OWNER/NAME NUMBER"
REPO="$2"
PR="$3"
require_tools gh jq
check_repo "$REPO"
case "$PR" in
  '' | *[!0-9]*) die "not a pull request number: $PR" ;;
esac
DIR="$1/prs/$REPO/$PR"
mkdir -p "$DIR"

# mergeCommit stays null until the PR merges, where REST's merge_commit_sha names a test merge.
gh pr view "$PR" -R "$REPO" \
  --json number,title,state,isDraft,mergedAt,mergeCommit,changedFiles,body,comments,reviews,url \
  > "$DIR/pr.json"
# The files endpoint lists up to 3000 paths; GitHub stops sending a diff past 300 files.
gh api --paginate "repos/$REPO/pulls/$PR/files?per_page=100" \
  --jq '.[] | [.status, .additions, .deletions, .filename] | @tsv' > "$DIR/files.txt"
if ! error=$({ gh pr diff "$PR" -R "$REPO" > "$DIR/diff.txt"; } 2>&1); then
  printf 'SKIPPED: diff -- %s; files.txt lists %s of its %s changed files\n' "${error%%$'\n'*}" \
    "$(($(wc -l < "$DIR/files.txt")))" "$(jq .changedFiles "$DIR/pr.json")" > "$DIR/diff.txt"
fi
{
  printf 'repo\tnumber\tstate\ttitle\turl\n'
  gh api --paginate "repos/$REPO/issues/$PR/timeline?per_page=100" --jq "$PR_STATE_JQ"'.[]
    | select(.event == "cross-referenced" and .source.issue.pull_request != null) | .source.issue
    | [.repository.full_name, .number, pr_state, .title, .html_url] | @tsv'
} > "$DIR/xrefs.tsv"

printf '%s\n' "$DIR"
