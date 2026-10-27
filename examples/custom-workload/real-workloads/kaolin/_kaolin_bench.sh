#!/usr/bin/env bash
# HyperloomHPC adapter — kaolin spherical-gaussian inner-product kernel on AMD GPUs.
#
# Workload source: https://github.com/AMD-Ecosystem/kaolin  branch `rocm-enablement`
# The kernel is JIT-built from source by benchmark.py (torch cpp_extension), so the
# loop's KERNEL phase can edit the .cu and this re-benchmarks it -- the same adapter
# closes the whole measure -> optimize -> validate loop.
#
# Env:
#   KAOLIN_SRC        kaolin checkout (default /work/kaolin)
#   KAOLIN_ROCM_INC   dir with rocThrust/hipsparse headers the slim image lacks
#   KAOLIN_REF_STORE  where the correctness reference is cached
set -euo pipefail

RESULT_DIR="${RESULT_DIR:-/tmp/hlhpc-kaolin}"; mkdir -p "$RESULT_DIR"
RESULT_FILENAME="${RESULT_FILENAME:-inferencex_result}"
LOG="$RESULT_DIR/benchmark.out"

export KAOLIN_SRC="${KAOLIN_SRC:-/work/kaolin}"
export PYTORCH_ROCM_ARCH="${PYTORCH_ROCM_ARCH:-gfx950}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-/tmp/kaolin_ext}"
export KAOLIN_REF_STORE="${KAOLIN_REF_STORE:-/tmp/kaolin_refs}"

# --- build-env fixes for the slim rocm/pytorch image (both are load-bearing) ---
# (1) it lacks rocThrust/hipsparse headers that torch's c10/util/complex.h needs
if [[ -d "${KAOLIN_ROCM_INC:-}" ]]; then
    export HIPCC_COMPILE_FLAGS_APPEND="-I${KAOLIN_ROCM_INC} ${HIPCC_COMPILE_FLAGS_APPEND:-}"
    export CPLUS_INCLUDE_PATH="${KAOLIN_ROCM_INC}:${CPLUS_INCLUDE_PATH:-}"
fi
# (2) it ships only versioned libamdhip64.so.N -- the linker needs the bare .so
_A=$(find /opt/venv -name 'libamdhip64.so*' 2>/dev/null | head -1)
if [[ -n "${_A:-}" ]]; then
    _D=$(dirname "$_A")
    [[ -e "$_D/libamdhip64.so" ]] || ln -sf "$(basename "$_A")" "$_D/libamdhip64.so" 2>/dev/null || true
    export LIBRARY_PATH="$_D:${LIBRARY_PATH:-}"
fi

set +e
python3 "${KAOLIN_BENCH_PY:-/work/kaolin_bench.py}" > "$LOG" 2>&1
RC=$?
set -e

GEO=$(grep -oE 'geomean_fwd_bwd_ms=[0-9.]+' "$LOG" | tail -1 | cut -d= -f2 || true)
PASS=$(grep -q 'RESULT correctness PASSED' "$LOG" && echo true || echo false)
REL=$(grep -oE 'max_rel=[0-9.e+-]+' "$LOG" | tail -1 | cut -d= -f2 || echo "nan")

fail() {
    printf '{"output_throughput": 0.0, "throughput_unit": "kops/s", "quality_gate": {"passed": false, "reason": "%s"}}\n' "$1" \
        > "$RESULT_DIR/$RESULT_FILENAME.json"
    echo "[kaolin] FAIL ($1); see $LOG" >&2; tail -15 "$LOG" >&2; exit 1
}
[[ $RC -ne 0 ]]      && fail "benchmark_exit_$RC"
[[ -z "${GEO:-}" ]]  && fail "no_objective_parsed"
[[ "$PASS" != true ]] && fail "correctness_failed"

TPUT=$(python3 -c "print(f'{1000.0/$GEO:.4f}')")
cat > "$RESULT_DIR/$RESULT_FILENAME.json" <<JSON
{
  "output_throughput": $TPUT,
  "throughput_unit": "kops/s (1000 / geomean_fwd_bwd_ms)",
  "mean_latency_ms": $GEO,
  "quality_gate": {
    "passed": $PASS,
    "max_rel": $REL,
    "metric": "kernel output matches the pre-optimization reference (rel < 1e-3) and is run-to-run deterministic"
  },
  "workload": {
    "name": "kaolin-sg-inner-product",
    "runner_type": "${RUNNER_TYPE:-}",
    "shapes": "64x128, 128x512, 256x1024 (fwd+bwd geomean)"
  }
}
JSON
echo "[kaolin] geomean=${GEO}ms -> $TPUT kops/s, gate passed=$PASS"
