#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# Step 3 of the check-prior-art skill: find the open pull requests that touch a planned path.
#
# usage: overlap.sh WORK
# Reads WORK/repos.txt and WORK/paths.txt, and writes WORK/overlap.tsv and WORK/scan.txt as Step 3
# of SKILL.md describes. An empty overlap.tsv means "no overlap" only where scan.txt says COMPLETE.

set -euo pipefail

# shellcheck disable=SC1091  # shellcheck checks lib.sh as a file of its own
. "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

[ "$#" -eq 1 ] || die "usage: overlap.sh WORK"
WORK="$1"
require_tools gh jq
mapfile -t REPOS < <(entries "$WORK/repos.txt")
[ "${#REPOS[@]}" -gt 0 ] || die "$WORK/repos.txt names no repository"
for repo in "${REPOS[@]}"; do
  check_repo "$repo"
done
stray=$(entries "$WORK/paths.txt" \
  | awk -F'\t' 'NR == FNR { known[$0] = 1; next } NF != 2 || !($1 in known)' \
    <(printf '%s\n' "${REPOS[@]}") -)
[ -z "$stray" ] || die "paths.txt lines are <owner/name><TAB><path> for a repository in repos.txt: $stray"

# An upstream repository has thousands of open PRs, and one gh pr list call returns all of
# them or nothing, so this pages GraphQL itself: a failed page keeps what was compared and
# names where the scan stopped. A page of 100 (the API's largest) takes about 5 s, so
# MAX_PAGES bounds one repository to under ten minutes.
# CREATED_AT keeps the order stable while the scan runs: under UPDATED_AT, a PR updated
# mid-scan would jump ahead of the cursor and never be compared.
PAGE_SIZE=100
MAX_PAGES=100
# shellcheck disable=SC2016  # $owner, $name and $cursor are GraphQL variables, not shell ones
QUERY='query($owner: String!, $name: String!, $cursor: String) {
  repository(owner: $owner, name: $name) {
    pullRequests(states: OPEN, first: '"$PAGE_SIZE"', after: $cursor,
                 orderBy: {field: CREATED_AT, direction: DESC}) {
      totalCount
      pageInfo { hasNextPage endCursor }
      nodes { number title url isDraft updatedAt changedFiles
              files(first: 100) { nodes { path changeType } } }
    }
  }
}'
# shellcheck disable=SC2016  # $want, $repo and $hit are jq variables, not shell ones
JQ_DEFS='
  def planned($want): . as $f
    | any($want[]; . as $w | if ($w | endswith("/")) then ($f | startswith($w)) else $f == $w end);
  def row($repo; $hit):
    [$repo, .number, "open", .isDraft, .title, .url, .updatedAt, ($hit | length), ($hit | join(" "))] | @tsv;
'

raw=$(mktemp -d)
trap 'rm -rf "$raw"' EXIT
printf 'repo\tnumber\tstate\tdraft\ttitle\turl\tupdated\toverlap\tpaths\n' > "$WORK/overlap.tsv"
: > "$WORK/scan.txt"

# Appends the page's PRs that list a planned path to overlap.tsv, and writes to undecided the
# ones GraphQL cannot settle: a file list cut at 100, or a renamed file, listed by its new
# path only.
compare_page() {
  local repo="$1" want="$2"
  jq -r --arg repo "$repo" --argjson want "$want" "$JQ_DEFS"'
    .data.repository.pullRequests.nodes[]
    | [(.files.nodes // [])[].path | select(planned($want))] as $hit
    | select($hit | length > 0)
    | row($repo; $hit)' "$raw/page.json" >> "$WORK/overlap.tsv"
  jq -r --argjson want "$want" "$JQ_DEFS"'
    .data.repository.pullRequests.nodes[]
    | (.files.nodes // []) as $files
    | select(all($files[]; .path | planned($want) | not))
    | select(.changedFiles > ($files | length) or any($files[]; .changeType == "RENAMED"))
    | .number' "$raw/page.json" > "$raw/undecided"
}

# Compares an undecided PR on its REST file list, which names a renamed file's old path and
# runs to 3000 files. A PR that list cannot settle gets a PARTIAL line of its own.
relist() {
  local repo="$1" want="$2" number="$3"
  if ! gh api --paginate "repos/$repo/pulls/$number/files?per_page=100" \
    > "$raw/files.json" 2> "$raw/error"; then
    printf 'PARTIAL: %s#%s: its full file list could not be read, so its overlap was not decided -- %s\n' \
      "$repo" "$number" "$(head -1 "$raw/error")" >> "$WORK/scan.txt"
    return
  fi
  jq -r --arg repo "$repo" --argjson want "$want" --argjson number "$number" \
    --slurpfile files "$raw/files.json" "$JQ_DEFS"'
    .data.repository.pullRequests.nodes[]
    | select(.number == $number)
    | ($files | add // []) as $listed
    | [$listed[] | if (.previous_filename // "" | planned($want)) then "\(.previous_filename)->\(.filename)"
                   elif (.filename | planned($want)) then .filename else empty end] as $hit
    | if ($hit | length > 0) then row($repo; $hit) | "row\t\(.)"
      elif ($listed | length) < .changedFiles
      then "cut\tPARTIAL: \($repo)#\(.number): changes \(.changedFiles) files, only its first \($listed | length) were compared"
      else empty end' "$raw/page.json" > "$raw/relisted"
  awk 'sub(/^row\t/, "")' "$raw/relisted" >> "$WORK/overlap.tsv"
  awk 'sub(/^cut\t/, "")' "$raw/relisted" >> "$WORK/scan.txt"
}

# Pages through a repository's open PRs and prints the repository's scan.txt line.
scan_repo() {
  local repo="$1" want="$2" cursor="" total="" compared=0 page=0 more=true
  local args open listed number undecided
  while [ "$more" = true ] && [ "$page" -lt "$MAX_PAGES" ]; do
    page=$((page + 1))
    args=(-f owner="${repo%%/*}" -f name="${repo#*/}" -f query="$QUERY")
    if [ -n "$cursor" ]; then
      args+=(-f cursor="$cursor")
    fi
    if ! gh api graphql "${args[@]}" > "$raw/page.json" 2> "$raw/error"; then
      echo "PARTIAL: $repo: the scan stopped after $compared of ${total:-an unknown number of} open PRs -- $(head -1 "$raw/error")"
      return
    fi
    compare_page "$repo" "$want"
    mapfile -t undecided < "$raw/undecided"
    for number in "${undecided[@]}"; do
      relist "$repo" "$want" "$number"
    done
    read -r open listed more cursor < <(jq -r '.data.repository.pullRequests
      | "\(.totalCount) \(.nodes | length) \(.pageInfo.hasNextPage) \(.pageInfo.endCursor // "")"' "$raw/page.json")
    total=${total:-$open}
    compared=$((compared + listed))
  done
  if [ "$more" = true ]; then
    echo "PARTIAL: $repo: only the newest $compared of $total open PRs were compared"
  else
    echo "COMPLETE: $repo: all $compared open PRs were compared against $(jq length <<< "$want") planned path(s)"
  fi
}

for repo in "${REPOS[@]}"; do
  want=$(entries "$WORK/paths.txt" | awk -F'\t' -v repo="$repo" '$1 == repo { print $2 }' \
    | jq -Rsc 'split("\n") | map(select(length > 0))')
  if [ "$want" = '[]' ]; then
    echo "SKIPPED: $repo: no planned path in paths.txt, so its open PRs were not compared" >> "$WORK/scan.txt"
  else
    scan_repo "$repo" "$want" >> "$WORK/scan.txt"
  fi
done

cat "$WORK/scan.txt"
printf 'overlapping open PRs: %s\n' "$(($(wc -l < "$WORK/overlap.tsv") - 1))"
printf '%s\n' "$WORK"
