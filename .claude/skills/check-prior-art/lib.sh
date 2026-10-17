# shellcheck shell=bash
# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# Helpers the check-prior-art scripts share. Sourced, never run.

die() {
  echo "${0##*/}: $1" >&2
  exit 1
}

require_tools() {
  local tool
  for tool in "$@"; do
    command -v "$tool" >/dev/null 2>&1 || die "$tool is required"
  done
}

# The lines of an input file an agent wrote, without blank lines and # comments.
entries() {
  [ -f "$1" ] || die "$1 is missing"
  grep -v -e '^[[:space:]]*$' -e '^[[:space:]]*#' "$1" || true
}

check_repo() {
  [[ "$1" =~ ^[A-Za-z0-9-]+/[A-Za-z0-9._-]+$ ]] || die "not an owner/name repository: $1"
}

# A pull request's state as the issues API reports it, where a merged PR is "closed".
# shellcheck disable=SC2034  # read by the scripts that source this file
PR_STATE_JQ='def pr_state: if .pull_request.merged_at then "merged" else .state end; '
