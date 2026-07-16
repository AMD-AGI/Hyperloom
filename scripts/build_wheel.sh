#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Build the Hyperloom wheel and optionally publish it to a GitHub release.
#
# Build only (default):
#   scripts/build_wheel.sh
# Build and publish to the release matching the package version:
#   scripts/build_wheel.sh --publish
#
# Publishing uses `gh` (which reuses the host's GitHub auth) and uploads the
# built wheel as a release asset with --clobber. The Hyperloom repo is
# private/internal, so the asset is only downloadable with authentication.

set -euo pipefail

_script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${_script_dir}/.." && pwd)"

PUBLISH=0
DRY_RUN=0
TAG=""
GH_REPO="AMD-AGI/Hyperloom"
OUT_DIR="${REPO_ROOT}/dist"

usage() {
  cat <<'EOF'
Usage: scripts/build_wheel.sh [options]

Build the Hyperloom wheel into dist/, and optionally publish it to a GitHub release.

Options:
  --publish        Upload the built wheel to a GitHub release (default: build only)
  --tag TAG        Release tag to upload to (default: v<major.minor> from the version)
  --repo OWNER/REPO GitHub repo for the release (default: AMD-AGI/Hyperloom)
  --out DIR        Output directory for the wheel (default: dist/)
  --dry-run        Print actions without building or uploading
  -h, --help       Show this help
EOF
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --publish) PUBLISH=1 ;;
    --tag) [ "$#" -ge 2 ] || { echo "ERROR: --tag requires a value" >&2; exit 2; }; shift; TAG="$1" ;;
    --repo) [ "$#" -ge 2 ] || { echo "ERROR: --repo requires a value" >&2; exit 2; }; shift; GH_REPO="$1" ;;
    --out) [ "$#" -ge 2 ] || { echo "ERROR: --out requires a value" >&2; exit 2; }; shift; OUT_DIR="$1" ;;
    --dry-run) DRY_RUN=1 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "ERROR: unknown option '$1'" >&2; usage >&2; exit 2 ;;
  esac
  shift
done

log() { echo "[build-wheel] $*"; }
die() { echo "[build-wheel ERROR] $*" >&2; exit 1; }

# Prefer the canonical ROCm venv when present, otherwise python3 on PATH.
if [ -x "/opt/venv/bin/python" ]; then
  PYTHON="/opt/venv/bin/python"
else
  PYTHON="$(command -v python3 || true)"
fi
[ -n "${PYTHON}" ] || die "no python interpreter found (set /opt/venv or python3 on PATH)"

# Read the package version from pyproject.toml for tag derivation / logging.
VERSION="$("${PYTHON}" - "${REPO_ROOT}/pyproject.toml" <<'PY'
import re, sys
text = open(sys.argv[1]).read()
m = re.search(r'^version\s*=\s*["\']([^"\']+)["\']', text, re.M)
print(m.group(1) if m else "")
PY
)"
[ -n "${VERSION}" ] || die "could not read version from pyproject.toml"
log "package version: ${VERSION}"

# Default release tag is v<major.minor> (matches the existing v0.8 releases).
if [ -z "${TAG}" ]; then
  _major_minor="$(printf '%s' "${VERSION}" | cut -d. -f1-2)"
  TAG="v${_major_minor}"
fi

# --- Build ---
if [ "${DRY_RUN}" -eq 1 ]; then
  log "would build: ${PYTHON} -m pip wheel --no-deps --no-build-isolation -w ${OUT_DIR} ${REPO_ROOT}"
else
  log "building wheel into ${OUT_DIR}"
  "${PYTHON}" -m pip wheel --no-deps --no-build-isolation -w "${OUT_DIR}" "${REPO_ROOT}"
fi

# Resolve the built wheel path for this version.
WHEEL="${OUT_DIR}/hyperloom_inference_optimizer-${VERSION}-py3-none-any.whl"
if [ "${DRY_RUN}" -eq 0 ]; then
  [ -f "${WHEEL}" ] || WHEEL="$(ls -1t "${OUT_DIR}"/hyperloom_inference_optimizer-*.whl 2>/dev/null | head -1 || true)"
  [ -n "${WHEEL}" ] && [ -f "${WHEEL}" ] || die "built wheel not found in ${OUT_DIR}"
  log "built: ${WHEEL}"
fi

# --- Publish (opt-in) ---
if [ "${PUBLISH}" -eq 0 ]; then
  log "build complete (pass --publish to upload to a GitHub release)"
  exit 0
fi

if [ "${DRY_RUN}" -eq 1 ]; then
  log "would publish: gh release upload ${TAG} ${WHEEL:-<wheel>} --clobber -R ${GH_REPO}"
  exit 0
fi

command -v gh >/dev/null 2>&1 || die "gh CLI not found; needed to publish. Install gh + 'gh auth login'."
gh auth status >/dev/null 2>&1 || die "gh is not authenticated; run 'gh auth login' (needs write access to ${GH_REPO})."
gh release view "${TAG}" -R "${GH_REPO}" >/dev/null 2>&1 \
  || die "release ${TAG} not found in ${GH_REPO}; create it first or pass --tag."

log "uploading ${WHEEL##*/} to ${GH_REPO}@${TAG} (--clobber)"
gh release upload "${TAG}" "${WHEEL}" --clobber -R "${GH_REPO}"
log "published: $(gh release view "${TAG}" -R "${GH_REPO}" --json assets \
  --jq ".assets[] | select(.name==\"${WHEEL##*/}\") | .url" 2>/dev/null || echo "${TAG}/${WHEEL##*/}")"
