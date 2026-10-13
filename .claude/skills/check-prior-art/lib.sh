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

# Replaces the file $1 with standard input in one rename, so a run killed midway never
# leaves it half written.
replace_file() {
  cat > "$1.tmp"
  mv -f "$1.tmp" "$1"
}
