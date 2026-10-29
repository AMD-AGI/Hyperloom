#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Build the hosting layer of a hosted Hyperloom run: a directory that any framework image runs as it is,
# in place of an application image built for Hyperloom (see Dockerfile, which stays for those who want one).
#
#   scripts/hosted_app_image/build_layer.sh --out /path/to/hosted-layer [--claude-version V] [--ca-file PEM]
#
# It creates <out>/<short sha of this checkout>-claude<version>/ holding
#   start.sh          the application's command: `sh <layer dir>/start.sh`
#   app_bootstrap.sh  the bootstrap start.sh execs
#   bin/claude        the Claude Code CLI (one self-contained binary; nothing is fetched when a run starts)
#   ca.pem            an extra CA certificate for the run's TLS (only with --ca-file)
# and refuses to overwrite a layer that exists. The directory is world-readable: publish it to the
# shared filesystem the application container mounts.

set -euo pipefail

_script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${_script_dir}/../.." && pwd)"

OUT=""
CLAUDE_VERSION="latest"
CA_FILE=""

usage() {
  cat <<'USAGE'
Usage: scripts/hosted_app_image/build_layer.sh --out DIR [options]

  --out DIR           Directory that holds the layers (required)
  --claude-version V  Claude Code version to install (default: latest; the layer name records what it was)
  --ca-file PEM       Extra CA certificate the run trusts
  -h, --help          Show this help
USAGE
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --out) [ "$#" -ge 2 ] || { echo "ERROR: --out requires a value" >&2; exit 2; }; shift; OUT="$1" ;;
    --claude-version) [ "$#" -ge 2 ] || { echo "ERROR: --claude-version requires a value" >&2; exit 2; }; shift; CLAUDE_VERSION="$1" ;;
    --ca-file) [ "$#" -ge 2 ] || { echo "ERROR: --ca-file requires a value" >&2; exit 2; }; shift; CA_FILE="$1" ;;
    -h|--help) usage; exit 0 ;;
    *) echo "ERROR: unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
  shift
done

[ -n "$OUT" ] || { echo "ERROR: --out is required" >&2; usage >&2; exit 2; }
[ -z "$CA_FILE" ] || grep -q "BEGIN CERTIFICATE" "$CA_FILE" || { echo "ERROR: $CA_FILE holds no PEM certificate" >&2; exit 2; }
command -v curl >/dev/null || { echo "ERROR: curl is required" >&2; exit 2; }

short_sha="$(git -C "$REPO_ROOT" rev-parse --short=8 HEAD)"
[ -z "$(git -C "$REPO_ROOT" status --porcelain --untracked-files=no)" ] || { echo "ERROR: the checkout has uncommitted changes; the layer name is its commit" >&2; exit 2; }

work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

# Claude Code's native installer, as install_kernel_tools.sh and the Dockerfile use it, into a scratch home.
mkdir -p "$work/home"
curl -fsSL --connect-timeout 20 https://claude.ai/install.sh | HOME="$work/home" bash -s -- "$CLAUDE_VERSION"
installed="$(readlink -f "$work/home/.local/bin/claude")"
[ -x "$installed" ] || { echo "ERROR: the installer left no claude binary" >&2; exit 1; }
version="$("$installed" --version | awk '{print $1; exit}')"
[[ "$version" =~ ^[0-9][0-9A-Za-z.+-]*$ ]] || { echo "ERROR: cannot read the claude version" >&2; exit 1; }

layer="$OUT/${short_sha}-claude${version}"
[ ! -e "$layer" ] || { echo "ERROR: $layer exists; layers are immutable" >&2; exit 1; }
mkdir -p "$OUT"
stage="$(mktemp -d "$OUT/.build.XXXXXX")"
trap 'rm -rf "$work" "$stage"' EXIT
mkdir "$stage/bin"
install -m 0755 "$installed" "$stage/bin/claude"
install -m 0755 "$REPO_ROOT/src/hyperloom/hosted/app_bootstrap.sh" "$stage/app_bootstrap.sh"
install -m 0755 "$REPO_ROOT/src/hyperloom/hosted/start.sh" "$stage/start.sh"
[ -z "$CA_FILE" ] || install -m 0644 "$CA_FILE" "$stage/ca.pem"
chmod 0755 "$stage"
"$stage/bin/claude" --version >/dev/null
mv -T "$stage" "$layer"
echo "$layer"
