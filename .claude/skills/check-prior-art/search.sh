#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# Step 2 of the check-prior-art skill: search each repository's open and merged pull requests
# for each search phrase.
#
# usage: search.sh WORK
# Reads WORK/repos.txt (owner/name per line) and WORK/terms.txt (one GitHub search phrase per
# line). Appends to WORK/queries.tsv, as each query is answered, how many results it listed out
# of how many matched, or that it failed and why; a failed query does not stop the rest, so
# coverage is read from queries.tsv, never from how short the result is. At the end, writes
# WORK/candidates.tsv: one row per pull request any query found, with the number of queries
# that found it.

set -euo pipefail

# shellcheck disable=SC1091  # shellcheck checks lib.sh as a file of its own
. "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

[ "$#" -eq 1 ] || die "usage: search.sh WORK"
WORK="$1"
require_tools gh
mapfile -t REPOS < <(entries "$WORK/repos.txt")
mapfile -t TERMS < <(entries "$WORK/terms.txt")
[ "${#REPOS[@]}" -gt 0 ] || die "$WORK/repos.txt names no repository"
[ "${#TERMS[@]}" -gt 0 ] || die "$WORK/terms.txt holds no search phrase"
for repo in "${REPOS[@]}"; do
  check_repo "$repo"
done

# One page of 100, the API's largest: a query that matches more is too broad to read, and its
# TRUNCATED row says so. The first line out is the query's status, then one line per PR.
# shellcheck disable=SC2016  # $n is a jq variable, not a shell one
JQ="$PR_STATE_JQ"'(.items | length) as $n
  | ([if .incomplete_results then "INCOMPLETE" elif $n < .total_count then "TRUNCATED" else "OK" end,
      $n, .total_count] | @tsv),
    (.items[] | [.number, pr_state, .draft, .title, .html_url, .updated_at] | @tsv)'

raw=$(mktemp -d)
trap 'rm -rf "$raw"' EXIT
: > "$raw/rows.tsv"
printf 'status\tlisted\ttotal\tquery\tnote\n' > "$WORK/queries.tsv"
for repo in "${REPOS[@]}"; do
  for state in open merged; do
    for terms in "${TERMS[@]}"; do
      query="repo:$repo is:pr is:$state $terms"
      if gh api -X GET search/issues -f q="$query" -f per_page=100 --jq "$JQ" > "$raw/page" 2> "$raw/error"; then
        printf '%s\t%s\t\n' "$(head -1 "$raw/page")" "$query" >> "$WORK/queries.tsv"
        awk -v repo="$repo" 'NR > 1 { print repo "\t" $0 }' "$raw/page" >> "$raw/rows.tsv"
      else
        printf 'FAILED\t-\t-\t%s\t%s\n' "$query" "$(head -1 "$raw/error")" >> "$WORK/queries.tsv"
      fi
      # The search API allows an authenticated user 30 queries a minute.
      sleep 2
    done
  done
done

# A candidate is one row however many queries found it: their count ranks the shortlist in
# Step 4, the most recently updated first among equals.
{
  printf 'repo\tnumber\tstate\tdraft\ttitle\turl\tupdated\thits\n'
  awk -F'\t' -v OFS='\t' '
    { key = $1 FS $2; if (!(key in hits)) { order[++n] = key; row[key] = $0 } hits[key]++ }
    END { for (i = 1; i <= n; i++) print row[order[i]], hits[order[i]] }
  ' "$raw/rows.tsv" | LC_ALL=C sort -t "$(printf '\t')" -k8,8nr -k7,7r -k1,1 -k2,2n
} > "$WORK/candidates.tsv"

awk -F'\t' 'NR > 1 { count[$1]++ } END { for (s in count) printf "%-10s %s query(ies)\n", s, count[s] }' \
  "$WORK/queries.tsv" | LC_ALL=C sort
printf '%-10s %s pull request(s)\n' candidates "$(($(wc -l < "$WORK/candidates.tsv") - 1))"
printf '%s\n' "$WORK"
