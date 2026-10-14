#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# Step 6 of the analyze-upstream-pr skill: check-prior-art's search and path scan find the pull
# request under analysis too, and its own diff changes every planned path, so it would be
# classified as its own duplicate.
#
# usage: drop_self.sh WORK OWNER/NAME NUMBER
# Removes OWNER/NAME#NUMBER from WORK/candidates.tsv and WORK/overlap.tsv, the tables
# check-prior-art's shortlist starts from, and prints how many rows it removed from each.

set -euo pipefail

# shellcheck disable=SC1091  # shellcheck checks lib.sh as a file of its own
. "$(dirname "${BASH_SOURCE[0]}")/../check-prior-art/lib.sh"

[ "$#" -eq 3 ] || die "usage: drop_self.sh WORK OWNER/NAME NUMBER"
WORK="$1"
REPO="$2"
PR="$3"
check_repo "$REPO"
case "$PR" in
  '' | *[!0-9]*) die "not a pull request number: $PR" ;;
esac
TABLES=(candidates.tsv overlap.tsv)
for table in "${TABLES[@]}"; do
  [ -f "$WORK/$table" ] || die "$WORK/$table is missing: run check-prior-art's search.sh and overlap.sh first"
done

# GitHub matches owner/name without regard to case.
for table in "${TABLES[@]}"; do
  before=$(wc -l < "$WORK/$table")
  awk -F'\t' -v repo="$REPO" -v pr="$PR" 'tolower($1) != tolower(repo) || $2 != pr' "$WORK/$table" \
    > "$WORK/$table.tmp"
  mv -f "$WORK/$table.tmp" "$WORK/$table"
  printf '%-14s %s row(s) removed\n' "$table" "$((before - $(wc -l < "$WORK/$table")))"
done
printf '%s\n' "$WORK"
