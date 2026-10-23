#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Start a hosted Hyperloom run inside a framework image that has no Hyperloom.
#
# The host runs this script as the application's command. It
#   1. reads the version to run from HYPERLOOM_SOURCE_REPO and HYPERLOOM_SOURCE_SHA (a full commit id),
#   2. builds the wheel for that commit once, into $USER_DATA_PATH/.hyperloom/wheels/<sha>/, and reuses it after,
#   3. creates a venv with --system-site-packages in $HYPERLOOM_STATE_DIR/venv and installs the dependencies from
#      the commit's requirements/app.lock with --require-hashes, then the wheel,
#   4. sets HYPERLOOM_CODE_REVISION=<sha> and runs `python -m hyperloom hosted run`.
#
# Nothing of the checkout stays behind: the wheel carries no .git, so a restart sees the same code version. An input the
# script cannot work with (no commit, a bad one, no lock file) ends the run with a final exit.json; a failure that may be
# transient (the network, the disk) exits without one, and the host restarts the run.

set -euo pipefail

state_dir="${HYPERLOOM_STATE_DIR:-}"

# A message here is a constant: it is written into JSON as it stands.
fail_final() {
  echo "hosted bootstrap: $1" >&2
  if [[ -n "$state_dir" && -d "$state_dir" ]]; then
    printf '{"final": true, "returncode": 2, "error": "%s"}\n' "$1" >"$state_dir/.exit.json.tmp"
    mv -f "$state_dir/.exit.json.tmp" "$state_dir/exit.json"
  fi
  exit 2
}

fail() {
  echo "hosted bootstrap: $1" >&2
  exit 1
}

[[ -n "$state_dir" && "$state_dir" == /* ]] || fail_final "HYPERLOOM_STATE_DIR is not an absolute path"
[[ "${USER_DATA_PATH:-}" == /* ]] || fail_final "USER_DATA_PATH is not an absolute path"

repo="${HYPERLOOM_SOURCE_REPO:-}"
sha="${HYPERLOOM_SOURCE_SHA:-}"
[[ -n "$repo" ]] || fail_final "HYPERLOOM_SOURCE_REPO is not set"
[[ "$sha" =~ ^[0-9a-f]{40}$ ]] || fail_final "HYPERLOOM_SOURCE_SHA must be a full 40-character commit id"

# OWNER/NAME on GitHub, or a git URL or path with a host or a root; anything else is an input error, not a network one.
url_pattern='^(https?|ssh)://[^/[:space:]]+/[^[:space:]]+$|^git@[^:[:space:]]+:[^[:space:]]+$|^file:///[^[:space:]]+$|^/[^[:space:]]+$'
if [[ "$repo" =~ $url_pattern ]]; then
  url="$repo"
elif [[ "$repo" =~ ^[A-Za-z0-9._-]+/[A-Za-z0-9._-]+$ ]]; then
  url="https://github.com/${repo}.git"
else
  fail_final "HYPERLOOM_SOURCE_REPO is neither OWNER/NAME nor a git URL"
fi

cache_root="$USER_DATA_PATH/.hyperloom"
wheel_dir="$cache_root/wheels/$sha"
venv="$state_dir/venv"
stamp="$venv/.hyperloom-install"
pip=("$venv/bin/python" -m pip --disable-pip-version-check --no-input)

work=""
cleanup() {
  [[ -z "$work" ]] || rm -rf "$work"
  work=""
}
trap cleanup EXIT

build_wheel() {
  local src
  mkdir -p "$cache_root/wheels"
  work="$(mktemp -d "$cache_root/build.XXXXXX")"
  src="$work/src"
  git init -q "$src"
  git -C "$src" fetch -q --depth 1 "$url" "$sha" || git -C "$src" fetch -q "$url" || fail "cannot fetch the source commit"
  # The repository answered (the fetch above succeeded) and has no such commit: nothing a restart can fix.
  git -C "$src" checkout -q --detach "$sha" || fail_final "the source repository has no such commit"
  [[ -f "$src/requirements/app.lock" ]] || fail_final "the source commit has no requirements/app.lock"
  "${pip[@]}" install -q --require-hashes --no-deps -r "$src/requirements/app.lock"
  mkdir "$work/out"
  "${pip[@]}" wheel -q --no-deps --no-build-isolation -w "$work/out" "$src"
  cp "$src/requirements/app.lock" "$work/out/app.lock"
  : >"$work/out/.complete"
  # The directory appears whole or not at all; a concurrent build of the same commit that won is as good as ours.
  mv -T "$work/out" "$wheel_dir" 2>/dev/null || [[ -f "$wheel_dir/.complete" ]] || fail "cannot store the wheel"
  cleanup
}

install_into_venv() {
  rm -rf "$venv"
  python3 -m venv --system-site-packages "$venv"
  if [[ -f "$wheel_dir/.complete" ]]; then
    "${pip[@]}" install -q --require-hashes --no-deps -r "$wheel_dir/app.lock"
  else
    build_wheel # leaves the pinned dependencies installed in the venv
  fi
  local wheels=("$wheel_dir"/*.whl)
  [[ ${#wheels[@]} -eq 1 && -f "${wheels[0]}" ]] || fail "expected exactly one wheel in $wheel_dir"
  "${pip[@]}" install -q --no-deps --no-index "${wheels[0]}"
  printf '%s\n' "$sha" >"$stamp"
}

if [[ "$(cat "$stamp" 2>/dev/null || true)" != "$sha" ]]; then
  install_into_venv
fi

export HYPERLOOM_CODE_REVISION="$sha"
export VIRTUAL_ENV="$venv"
export PATH="$venv/bin:$PATH"
exec "$venv/bin/python" -m hyperloom hosted run
