#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Start a hosted Hyperloom run inside a framework image that has no Hyperloom.
#
# The host runs this script as the application's command. It
#   1. reads the version to run from HYPERLOOM_SOURCE_REPO and HYPERLOOM_SOURCE_SHA (a full commit id),
#   2. checks that this account can write the state directory and USER_DATA_PATH: the legs of a run may be different
#      accounts of one group on the shared storage, and a directory an earlier leg left without group write ends the
#      run here, naming it, instead of at its first write,
#   3. builds the wheel for that commit in a directory of its own on the container's disk (never on the shared storage,
#      where another account of the group could put a wheel of its own in its place before the install) and installs
#      that wheel, and nothing else, into the framework's own Python environment (the one the legacy flow's
#      install.sh installs into, with the image's torch and ROCm stack), not into a venv of its own: a venv made from
#      it cannot see that stack. The runtime dependencies are not installed here: the entry runs the wheel's install.sh
#      before optimize, which resolves them in that environment as the legacy launchers do,
#   4. sets HYPERLOOM_CODE_REVISION=<sha> and runs `python -m hyperloom hosted run` with that Python.
#
# The environment records the installed commit, so a restart in the same container installs nothing; a new container
# builds the wheel again.
#
# A source on this machine (a path or a file:// URL, such as the host's mirror) may belong to another account: in a
# container that maps one id, a mirror the host's root owns shows as 65534. git refuses to fetch from a repository
# another account owns ("detected dubious ownership") unless safe.directory names it in the system or global config,
# and a -c does not reach the git-upload-pack the local transport starts (git drops it from that environment). So that
# fetch, and only it, runs with a global config in the container's temporary directory that trusts the source's
# resolved directory (and its .git) by exact path and nothing else, and names the source by that path; every other
# repository keeps git's ownership check.
#
# Nothing of the checkout stays behind: the wheel carries no .git, so a restart sees the same code version. An input the
# script cannot work with (no commit, a bad one) ends the run with a final exit.json; a failure that may be
# transient (the network, the disk) exits without one, and the host restarts the run.

set -euo pipefail

state_dir="${HYPERLOOM_STATE_DIR:-}"

# $1 as the inside of a JSON string.
json_text() {
  local text="${1//\\/\\\\}"
  text="${text//\"/\\\"}"
  printf '%s' "${text//[[:cntrl:]]/?}"
}

# exit.json is written beside its place, made new (set -C) under a name of this process's own that no one can guess
# (mktemp -u) with mode 0664 (umask 002, as the entry writes it), and renamed over it (mv -T, never into a directory
# of that name): a file another leg's account left there can be replaced, never rewritten. When the directory takes no
# file, stderr is all there is.
fail_final() {
  local written=""
  echo "hosted bootstrap: $1" >&2
  if [[ -n "$state_dir" && -d "$state_dir" ]]; then
    { written="$(mktemp -u "$state_dir/.exit.json.XXXXXXXXXX")" &&
      (umask 002 && set -C && printf '{"final": true, "returncode": 2, "error": "%s"}\n' "$(json_text "$1")" >"$written") &&
      mv -fT -- "$written" "$state_dir/exit.json"; } 2>/dev/null || {
      [[ -z "$written" ]] || rm -f -- "$written"
      echo "hosted bootstrap: cannot write $state_dir/exit.json" >&2
    }
  fi
  exit 2
}

# Whether this account can create and remove a file in the directory $2 (named $1); the run ends, naming it, if not.
require_writable() {
  local probe="$2/.hl-writable-$$.$RANDOM"
  { (set -C && : >"$probe") && rm -f -- "$probe"; } 2>/dev/null ||
    fail_final "$1 $2 cannot be written by this account (uid $EUID)"
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

# The framework Python, chosen as install.sh's resolve_python does (that script is in the wheel, which is not
# installed yet): the image's /opt/venv first, then an executable $PYTHON, then python3 on PATH.
if [[ "${INFERENCE_OPTIMIZER_FORCE_PYTHON:-0}" != "1" && -x /opt/venv/bin/python ]]; then
  framework_py=/opt/venv/bin/python
elif [[ -n "${PYTHON:-}" && -x "${PYTHON}" ]]; then
  framework_py="$PYTHON"
elif framework_py="$(command -v python3)"; then
  :
else
  fail_final "no framework Python was found (no /opt/venv/bin/python, PYTHON or python3)"
fi

require_writable HYPERLOOM_STATE_DIR "$state_dir"
(umask 002 && mkdir -p -- "$USER_DATA_PATH") 2>/dev/null || true # one that cannot be made is named next
require_writable USER_DATA_PATH "$USER_DATA_PATH"

# What the install of this environment holds, kept in the environment: a new container starts without it.
env_prefix="$("$framework_py" -c 'import sys; print(sys.prefix)')"
in_venv="$("$framework_py" -c 'import sys; print(int(sys.prefix != sys.base_prefix))')"
stamp="$env_prefix/.hyperloom-install"
pip=("$framework_py" -m pip --disable-pip-version-check --no-input)
# Outside a virtual environment pip refuses to modify the interpreter's packages without this (install.sh does the same).
if [[ "$in_venv" == "0" ]] && "$framework_py" -m pip install --break-system-packages --help >/dev/null 2>&1; then
  pip+=(--break-system-packages)
fi

work=""
trusted=""
cleanup() {
  [[ -z "$work" ]] || rm -rf "$work"
  [[ -z "$trusted" ]] || rm -f "$trusted"
  work=""
  trusted=""
}
trap cleanup EXIT

# For a source directory on this machine: a global config that trusts exactly that directory by its resolved path, and
# that path as the URL the fetch names. Any other source is fetched as it stands, with no config of ours.
fetch_url="$url"
trust_local_source() {
  local git_dir="$1" path="$url" scheme="" resolved
  if [[ "$path" == file://* ]]; then
    scheme="file://"
    path="${path#file://}"
  fi
  [[ "$path" == /* ]] || return 0
  # A source that is not a directory here is left to the fetch, which fails as it did before.
  resolved="$(cd -- "$path" 2>/dev/null && pwd -P)" || return 0
  trusted="$(mktemp "${TMPDIR:-/tmp}/hyperloom-source-trust.XXXXXX")"
  # Written through the new clone's git dir, so git looks for no repository in the working directory.
  git --git-dir="$git_dir" config --file "$trusted" --add safe.directory "$resolved"
  git --git-dir="$git_dir" config --file "$trusted" --add safe.directory "$resolved/.git"
  fetch_url="$scheme$resolved"
}

# The wheel of the commit, built in a new directory of this account's (mode 0700) on the container's own disk.
build_wheel() {
  local src
  work="$(mktemp -d "${TMPDIR:-/tmp}/hyperloom-wheel.XXXXXX")" || fail "cannot make a build directory"
  src="$work/src"
  git init -q "$src"
  trust_local_source "$src/.git"
  local fetch=(git -C "$src" fetch -q)
  [[ -z "$trusted" ]] || fetch=(env "GIT_CONFIG_GLOBAL=$trusted" "${fetch[@]}")
  "${fetch[@]}" --depth 1 "$fetch_url" "$sha" || "${fetch[@]}" "$fetch_url" || fail "cannot fetch the source commit"
  # The repository answered (the fetch above succeeded) and has no such commit: nothing a restart can fix.
  git -C "$src" checkout -q --detach "$sha" || fail_final "the source repository has no such commit"
  mkdir "$work/out"
  "${pip[@]}" wheel -q --no-deps -w "$work/out" "$src"
}

install_into_framework() {
  build_wheel
  local wheels=("$work/out"/*.whl)
  [[ ${#wheels[@]} -eq 1 && -f "${wheels[0]}" ]] || fail "expected exactly one wheel in $work/out"
  # Reinstall: another commit's wheel has the same version and pip would keep it.
  "${pip[@]}" install -q --force-reinstall --no-deps --no-index "${wheels[0]}"
  printf '%s\n' "$sha" >"$stamp"
  cleanup
}

if [[ "$(cat "$stamp" 2>/dev/null || true)" != "$sha" ]]; then
  install_into_framework
fi

export HYPERLOOM_CODE_REVISION="$sha"
if [[ "$in_venv" == "1" ]]; then
  export VIRTUAL_ENV="$env_prefix"
fi
framework_bin="$(dirname "$framework_py")"
export PATH="$framework_bin:$PATH"
exec "$framework_py" -m hyperloom hosted run
