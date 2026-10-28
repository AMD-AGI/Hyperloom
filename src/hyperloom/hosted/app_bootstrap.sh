#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Start a hosted Hyperloom run inside a framework image that has no Hyperloom.
#
# The host runs this script as the application's command. It
#   1. reads the version to run from HYPERLOOM_SOURCE_REPO and HYPERLOOM_SOURCE_SHA (a full commit id),
#   2. builds the wheel for that commit once, into $USER_DATA_PATH/.hyperloom/wheels/<sha>/, and reuses it after,
#   3. installs that wheel, and nothing else, into the framework's own Python environment (the one the legacy flow's
#      install.sh installs into, with the image's torch and ROCm stack), not into a venv of its own: a venv made from
#      it cannot see that stack. The runtime dependencies are not installed here: the entry runs the wheel's install.sh
#      before optimize, which resolves them in that environment as the legacy launchers do,
#   4. sets HYPERLOOM_CODE_REVISION=<sha> and runs `python -m hyperloom hosted run` with that Python.
#
# Nothing of the checkout stays behind: the wheel carries no .git, so a restart sees the same code version. An input the
# script cannot work with (no commit, a bad one) ends the run with a final exit.json; a failure that may be
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

cache_root="$USER_DATA_PATH/.hyperloom"
wheel_dir="$cache_root/wheels/$sha"
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
  mkdir "$work/out"
  "${pip[@]}" wheel -q --no-deps -w "$work/out" "$src"
  : >"$work/out/.complete"
  # The directory appears whole or not at all; a concurrent build of the same commit that won is as good as ours.
  mv -T "$work/out" "$wheel_dir" 2>/dev/null || [[ -f "$wheel_dir/.complete" ]] || fail "cannot store the wheel"
  cleanup
}

install_into_framework() {
  [[ -f "$wheel_dir/.complete" ]] || build_wheel
  local wheels=("$wheel_dir"/*.whl)
  [[ ${#wheels[@]} -eq 1 && -f "${wheels[0]}" ]] || fail "expected exactly one wheel in $wheel_dir"
  # Reinstall: another commit's wheel has the same version and pip would keep it.
  "${pip[@]}" install -q --force-reinstall --no-deps --no-index "${wheels[0]}"
  printf '%s\n' "$sha" >"$stamp"
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
