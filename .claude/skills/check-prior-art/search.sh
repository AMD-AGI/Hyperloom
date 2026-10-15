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
require_tools gh jq
mapfile -t REPOS < <(entries "$WORK/repos.txt")
mapfile -t TERMS < <(entries "$WORK/terms.txt")
[ "${#REPOS[@]}" -gt 0 ] || die "$WORK/repos.txt names no repository"
[ "${#TERMS[@]}" -gt 0 ] || die "$WORK/terms.txt holds no search phrase"
for repo in "${REPOS[@]}"; do
  check_repo "$repo"
done

# One page, the API's largest: a query that matches more is too broad to read, and its
# TRUNCATED row says so rather than spending the search quota on later pages.
PER_PAGE=100

raw=$(mktemp -d)
trap 'rm -rf "$raw"' EXIT
: > "$raw/rows.tsv"
printf 'status\tlisted\ttotal\tquery\tnote\n' > "$WORK/queries.tsv"

# The search API allows a few dozen queries a minute. GET /rate_limit can lag behind the
# search resource, so the last response's own headers decide whether the next query waits.
wait_for_quota() {
  local remaining reset pause
  remaining=$(awk -F': ' 'tolower($1) == "x-ratelimit-remaining" { print $2 + 0 }' "$raw/headers")
  reset=$(awk -F': ' 'tolower($1) == "x-ratelimit-reset" { print $2 + 0 }' "$raw/headers")
  if [ "${remaining:-1}" -lt 1 ] && [ -n "$reset" ]; then
    pause=$((reset - $(date +%s) + 1))
    if [ "$pause" -gt 0 ]; then
      sleep "$pause"
    fi
  fi
}

# gh -i prints the status line and headers, a blank line, then the body.
search() {
  local status=0
  gh api -i -X GET search/issues -f q="$1" -f per_page="$PER_PAGE" \
    > "$raw/response" 2> "$raw/error" || status=$?
  : > "$raw/headers"
  : > "$raw/page.json"
  tr -d '\r' < "$raw/response" | awk -v headers="$raw/headers" -v body="$raw/page.json" '
    in_body { print > body; next }
    /^$/ { in_body = 1; next }
    { print > headers }'
  return "$status"
}

for repo in "${REPOS[@]}"; do
  for state in open merged; do
    for terms in "${TERMS[@]}"; do
      query="repo:$repo is:pr is:$state $terms"
      if search "$query"; then
        jq -r --arg repo "$repo" '.items[]
          | [$repo, .number, (if .pull_request.merged_at then "merged" else .state end), .draft,
             .title, .html_url, .updated_at] | @tsv' "$raw/page.json" >> "$raw/rows.tsv"
        jq -r --arg query "$query" '
          [(if .incomplete_results then "INCOMPLETE"
            elif (.items | length) < .total_count then "TRUNCATED" else "OK" end),
           (.items | length), .total_count, $query, ""] | @tsv' "$raw/page.json" >> "$WORK/queries.tsv"
      else
        printf 'FAILED\t-\t-\t%s\t%s\n' "$query" "$(head -1 "$raw/error")" >> "$WORK/queries.tsv"
      fi
      wait_for_quota
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
