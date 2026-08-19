#!/usr/bin/env bash
# Regression: baremetal ROCm kernel path selection for legacy vs v0.5.17+ layouts.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INSTALL_SH="${SCRIPT_DIR}/install_baremetal.sh"
FIXTURE_ROOT="${SCRIPT_DIR}/../tests/fixtures/sglang_kernel_layouts"

fail() {
  echo "[test] FAIL: $*" >&2
  exit 1
}

pass() {
  echo "[test] PASS: $*"
}

eval "$(sed -n '/^sglang_kernel_rocm_build_dir()/,/^}/p' "$INSTALL_SH")"

legacy="${FIXTURE_ROOT}/legacy"
aot="${FIXTURE_ROOT}/v0517_aot"
empty="${FIXTURE_ROOT}/empty"

[ -f "${legacy}/sgl-kernel/setup_rocm.py" ] || fail "legacy fixture missing"
[ -f "${aot}/python/sglang/kernels/aot/setup_rocm.py" ] || fail "aot fixture missing"
[ ! -f "${empty}/sgl-kernel/setup_rocm.py" ] || fail "empty fixture should lack kernel"

dir="$(sglang_kernel_rocm_build_dir "$legacy")"
[ "$dir" = "${legacy}/sgl-kernel" ] || fail "legacy path got ${dir}"
pass "legacy -> sgl-kernel/"

dir="$(sglang_kernel_rocm_build_dir "$aot")"
[ "$dir" = "${aot}/python/sglang/kernels/aot" ] || fail "aot path got ${dir}"
pass "v0.5.17+ -> python/sglang/kernels/aot/"

sglang_kernel_rocm_build_dir "$empty" >/dev/null 2>&1 && fail "empty layout should fail"
pass "missing layout returns non-zero"

pass "sglang-kernel layout regression checks"
