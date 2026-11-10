#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# Find the internal-only references rule X9 (rules.md) keeps out of public text: an internal discussion-board link,
# topic or comment id, an internal host name or cluster path, a session or task id. X9's evidence step runs this over a
# PR's title, description, commit messages and added lines; check-card.sh runs it over the card before it is posted.
#
# usage: internal-refs.sh FILE...
#   prints '<file>:<line>: <kind>' for each hit and exits 1 when there is one, 2 when a FILE cannot be read. It names
#   the kind, never the reference, so the output can go where the reference itself may not: a public CI log, a commit
#   message.

set -euo pipefail

[ "$#" -ge 1 ] || {
  echo "usage: internal-refs.sh FILE..."
  exit 2
}

kinds=("board link" "board topic" "board comment" "cluster host" "cluster path" "session id" "task id")
patterns=(
  'primus-safe\.amd\.com/board'
  '\bboards/[A-Za-z0-9_.-]+'
  '\bc[0-9]{4,5}\b'
  '[A-Za-z0-9-]+\.crusoe\.amd\.com|\bm2m-v[0-9]+-[0-9]+\b|\bcrsuse[0-9]*-[a-z0-9-]+'
  '/shared_nfs\b'
  '\bMB-[0-9]{8}T[0-9]{6}Z'
  '\bktsk_[0-9A-Za-z]{10,}'
)

status=0
for file in "$@"; do
  if [ ! -r "$file" ] || [ -d "$file" ]; then
    echo "internal-refs.sh: cannot read ${file}" >&2
    exit 2
  fi
  for i in "${!patterns[@]}"; do
    while IFS=: read -r line _; do
      echo "${file}:${line}: ${kinds[$i]}"
      status=1
    done < <(grep -nE -- "${patterns[$i]}" "$file" || true)
  done
done
exit "$status"
