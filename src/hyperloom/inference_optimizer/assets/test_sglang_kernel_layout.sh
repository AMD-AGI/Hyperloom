#!/usr/bin/env bash
# Regression: SGLang v0.5.17+ has no in-tree sgl-kernel/; installer must pick pip path.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INSTALL_SH="${SCRIPT_DIR}/install_baremetal.sh"

fail() {
  echo "[test] FAIL: $*" >&2
  exit 1
}

pass() {
  echo "[test] PASS: $*"
}

eval "$(sed -n '/^resolve_sglang_kernel_pip_version()/,/^}/p' "$INSTALL_SH")"

if [ ! -d /tmp/sglang-517/.git ]; then
  git clone --depth 1 --branch v0.5.17 https://github.com/sgl-project/sglang.git /tmp/sglang-517
fi
if [ ! -d /tmp/sglang-516/.git ]; then
  git clone --depth 1 --branch v0.5.16 https://github.com/sgl-project/sglang.git /tmp/sglang-516
fi

[ -f /tmp/sglang-517/python/pyproject.toml ] || fail "v0.5.17 checkout missing pyproject.toml"
[ ! -f /tmp/sglang-517/sgl-kernel/setup_rocm.py ] || fail "v0.5.17 should not have in-tree sgl-kernel"
[ -f /tmp/sglang-516/sgl-kernel/setup_rocm.py ] || fail "v0.5.16 should have in-tree sgl-kernel"

grep -q 'install_sglang_kernel_rocm' "$INSTALL_SH" || fail "install_sglang_kernel_rocm helper missing"
grep -q 'no in-tree sgl-kernel/' "$INSTALL_SH" || fail "pip fallback log line missing"

ver="$(resolve_sglang_kernel_pip_version /tmp/sglang-517)"
[ "$ver" = "0.4.5" ] || fail "expected sglang-kernel 0.4.5 for v0.5.17, got ${ver}"
pass "resolve_sglang_kernel_pip_version v0.5.17 -> ${ver}"

export SGLANG_KERNEL_VERSION="9.9.9"
ver="$(resolve_sglang_kernel_pip_version /tmp/sglang-517)"
[ "$ver" = "9.9.9" ] || fail "SGLANG_KERNEL_VERSION override ignored"
unset SGLANG_KERNEL_VERSION
pass "SGLANG_KERNEL_VERSION override"

pass "sglang-kernel layout regression checks"
