#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Build the application image of a hosted Hyperloom run.
#
#   scripts/hosted_app_image/build.sh --tag registry.example/hyperloom-app:v1 [--base IMAGE] [--push]
#
# The image holds the framework base image, the bootstrap script and the specialist account;
# the Hyperloom code is built from HYPERLOOM_SOURCE_SHA when a run starts, not here.

set -euo pipefail

_script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${_script_dir}/../.." && pwd)"

TAG=""
BASE=""
PUSH=0

usage() {
  cat <<'USAGE'
Usage: scripts/hosted_app_image/build.sh --tag TAG [options]

  --tag TAG     Image reference to build (required)
  --base IMAGE  Framework base image (default: the Dockerfile's BASE_IMAGE)
  --push        Push the image after the build
  -h, --help    Show this help
USAGE
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --tag) [ "$#" -ge 2 ] || { echo "ERROR: --tag requires a value" >&2; exit 2; }; shift; TAG="$1" ;;
    --base) [ "$#" -ge 2 ] || { echo "ERROR: --base requires a value" >&2; exit 2; }; shift; BASE="$1" ;;
    --push) PUSH=1 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "ERROR: unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
  shift
done

[ -n "$TAG" ] || { echo "ERROR: --tag is required" >&2; usage >&2; exit 2; }

args=(build --file "${_script_dir}/Dockerfile" --tag "$TAG")
[ -z "$BASE" ] || args+=(--build-arg "BASE_IMAGE=${BASE}")
docker "${args[@]}" "$REPO_ROOT"
[ "$PUSH" -eq 0 ] || docker push "$TAG"
