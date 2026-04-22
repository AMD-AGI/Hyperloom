#!/usr/bin/env bash
# =============================================================================
# detect_project.sh — heuristic project classifier
#
# Usage: detect_project.sh <repo_root>
# Output (stdout): KEY=VALUE lines suitable for `source`.
#
# Sets:
#   PROJECT_CLASS    - hip-cmake-bench / pytorch-script / triton-collection / hpc-app / unknown
#   BUILD_SYSTEM     - cmake / setup.py / pyproject / make / none
#   BUILD_COMMAND    - exact command to (re)build from scratch
#   BENCH_COMMAND    - exact command to run the benchmark
#   BENCH_METRIC     - name of the metric (ms_per_iter / items_per_sec / etc.)
#   BENCH_METRIC_REGEX - regex with one capture group to grep the metric out of bench output
#   METRIC_LOWER_IS_BETTER - true / false
#   TEST_COMMAND     - command to run tests (may be empty)
#   CORRECTNESS_MODE - tests / golden-output / none
# =============================================================================

# NOTE: pipefail is intentionally OFF — many `find ... | head -1` pipelines below
# would otherwise SIGPIPE-fail and abort detection.
set -eu
REPO_ROOT="${1:?repo root required}"
[ -d "$REPO_ROOT" ] || { echo "ERROR: $REPO_ROOT not found" >&2; exit 1; }

cd "$REPO_ROOT"

# --- 1. Build system detection -------------------------------------------
HAS_CMAKE=$([ -f CMakeLists.txt ] || [ -f cpp/CMakeLists.txt ] && echo 1 || echo 0)
HAS_PYPROJECT=$([ -f pyproject.toml ] && echo 1 || echo 0)
HAS_SETUP_PY=$([ -f setup.py ] && echo 1 || echo 0)
HAS_MAKEFILE=$([ -f Makefile ] && echo 1 || echo 0)
HAS_BUILD_SH=$([ -f build.sh ] && echo 1 || echo 0)

# --- 2. Kernel language signals ------------------------------------------
HAS_HIP=$( find . -name '*.hip' -o -name '*.cuh' -o -name '*.cu' 2>/dev/null | head -1 | wc -l )
HAS_TRITON=$( grep -r -l --include='*.py' '@triton.jit' . 2>/dev/null | head -1 | wc -l )

# --- 3. Benchmark harness signals ----------------------------------------
HAS_GBENCH=$( grep -r -l --include='CMakeLists.txt' 'benchmark::benchmark' . 2>/dev/null | head -1 | wc -l )
HAS_CATCH2=$( grep -r -l --include='CMakeLists.txt' 'Catch2' . 2>/dev/null | head -1 | wc -l )
HAS_BENCH_DIR=$( [ -d bench ] || [ -d benchmarks ] || [ -d cpp/bench ] && echo 1 || echo 0 )
HAS_PY_BENCH=$( find . -maxdepth 3 -name 'bench*.py' -o -name 'benchmark*.py' 2>/dev/null | head -1 | wc -l )

# --- 4. Test harness signals ---------------------------------------------
# Locate the actual CTestTestfile.cmake (its dir is where ctest must be invoked)
CTEST_FILE=$(find build* cpp/build* -maxdepth 2 -name 'CTestTestfile.cmake' 2>/dev/null | head -1 || true)
HAS_CTEST=$([ -n "$CTEST_FILE" ] && echo 1 || echo 0)
CTEST_DIR=$([ -n "$CTEST_FILE" ] && dirname "$CTEST_FILE" || echo "")
HAS_PYTEST=$( ([ -d tests ] || [ -d test ] || grep -q '^pytest' pyproject.toml 2>/dev/null) && echo 1 || echo 0 )

# Determine the cmake source root (./ vs cpp/) so build/test commands point at the right place
if [ -f cpp/CMakeLists.txt ]; then
    CMAKE_SRC="cpp"
    DEFAULT_BUILD_DIR="cpp/build"
else
    CMAKE_SRC="."
    DEFAULT_BUILD_DIR="build"
fi

# --- 5. Classify ---------------------------------------------------------
PROJECT_CLASS="unknown"
BUILD_SYSTEM="none"
BUILD_COMMAND=""
BENCH_COMMAND=""
BENCH_METRIC="ms_per_iter"
BENCH_METRIC_REGEX="ms_per_iter[:\s=]+([0-9.]+)"
METRIC_LOWER_IS_BETTER="true"
TEST_COMMAND=""
CORRECTNESS_MODE="none"

if [ "$HAS_CMAKE" = 1 ] && [ "$HAS_HIP" -gt 0 ] && { [ "$HAS_GBENCH" -gt 0 ] || [ "$HAS_BENCH_DIR" = 1 ]; }; then
    PROJECT_CLASS="hip-cmake-bench"
    BUILD_SYSTEM="cmake"

    # Prefer build.sh if it exists (often handles ROCm specifics)
    if [ "$HAS_BUILD_SH" = 1 ]; then
        BUILD_COMMAND="bash $REPO_ROOT/build.sh"
    else
        BUILD_COMMAND="cmake -S $CMAKE_SRC -B $DEFAULT_BUILD_DIR -DCMAKE_BUILD_TYPE=Release && cmake --build $DEFAULT_BUILD_DIR -j\$(nproc)"
    fi

    # Find the first bench binary; user can override via BENCH_COMMAND
    BENCH_BIN=$(find $DEFAULT_BUILD_DIR -maxdepth 6 -type f -executable 2>/dev/null | grep -iE 'bench|benchmark' | head -1 || true)
    if [ -n "$BENCH_BIN" ]; then
        BENCH_COMMAND="$BENCH_BIN --benchmark_format=json --benchmark_min_time=2s --benchmark_repetitions=3"
        # Google Benchmark prints "real_time" in JSON; the regex picks the mean
        BENCH_METRIC="real_time_ns"
        BENCH_METRIC_REGEX='"real_time"\s*:\s*([0-9.]+)'
    else
        # No built binary yet — leave BENCH_COMMAND empty; emit a separate hint
        # variable so sourcing detected.env doesn't fail under `set -u`.
        BENCH_DIR=$(find . cpp -maxdepth 3 -type d \( -name bench -o -name benchmarks \) 2>/dev/null | head -1 || true)
        if [ -n "$BENCH_DIR" ]; then
            BENCH_HINT="rerun detect after build; expected bench dir: ${BENCH_DIR#./}"
            BENCH_METRIC="real_time_ns"
            BENCH_METRIC_REGEX='"real_time"\s*:\s*([0-9.]+)'
        fi
    fi

    if [ "$HAS_CTEST" = 1 ]; then
        # ctest must run from the dir containing CTestTestfile.cmake
        TEST_COMMAND="cd $CTEST_DIR && ctest --output-on-failure -j\$(nproc)"
        CORRECTNESS_MODE="tests"
    else
        CORRECTNESS_MODE="golden-output"
    fi

elif [ "$HAS_TRITON" -gt 0 ] && { [ "$HAS_PY_BENCH" -gt 0 ] || [ -d benchmarks ]; }; then
    PROJECT_CLASS="triton-collection"
    BUILD_SYSTEM=$([ "$HAS_PYPROJECT" = 1 ] && echo pyproject || echo setup.py)
    BUILD_COMMAND="pip install -e ."

    BENCH_PY=$(find . -maxdepth 3 \( -name 'bench*.py' -o -name 'benchmark*.py' \) 2>/dev/null | head -1)
    BENCH_COMMAND="python $BENCH_PY"
    BENCH_METRIC_REGEX="(?:ms_per_iter|latency_ms|time_ms)[:\s=]+([0-9.]+)"

    if [ "$HAS_PYTEST" = 1 ]; then
        TEST_COMMAND="pytest -x"
        CORRECTNESS_MODE="tests"
    fi

elif { [ "$HAS_PYPROJECT" = 1 ] || [ "$HAS_SETUP_PY" = 1 ]; } && grep -r -l --include='*.py' '^import torch\|^from torch' . 2>/dev/null | head -1 >/dev/null; then
    PROJECT_CLASS="pytorch-script"
    BUILD_SYSTEM=$([ "$HAS_PYPROJECT" = 1 ] && echo pyproject || echo setup.py)
    BUILD_COMMAND="pip install -e ."

    # Look for an obvious entrypoint
    ENTRY=$(find . -maxdepth 3 \( -name 'bench*.py' -o -name 'benchmark*.py' -o -name 'run.py' -o -name 'main.py' \) 2>/dev/null | head -1)
    if [ -n "$ENTRY" ]; then
        BENCH_COMMAND="python $ENTRY"
    fi
    BENCH_METRIC_REGEX="(?:ms_per_iter|samples_per_sec|tokens_per_sec)[:\s=]+([0-9.]+)"

    if [ "$HAS_PYTEST" = 1 ]; then
        TEST_COMMAND="pytest -x"
        CORRECTNESS_MODE="tests"
    fi

elif [ "$HAS_MAKEFILE" = 1 ] && [ "$HAS_HIP" -gt 0 ]; then
    PROJECT_CLASS="hpc-app"
    BUILD_SYSTEM="make"
    BUILD_COMMAND="make -j\$(nproc)"
    # User MUST provide BENCH_COMMAND override
fi

# --- 6. Emit -------------------------------------------------------------
# All values are quoted; the file must be sourceable under `set -euo pipefail`
# (no unescaped $VAR references in the values).
BENCH_HINT="${BENCH_HINT:-}"
cat <<OUT
PROJECT_CLASS=$PROJECT_CLASS
BUILD_SYSTEM=$BUILD_SYSTEM
BUILD_COMMAND="$BUILD_COMMAND"
BENCH_COMMAND="$BENCH_COMMAND"
BENCH_HINT="$BENCH_HINT"
BENCH_METRIC=$BENCH_METRIC
BENCH_METRIC_REGEX='$BENCH_METRIC_REGEX'
METRIC_LOWER_IS_BETTER=$METRIC_LOWER_IS_BETTER
TEST_COMMAND="$TEST_COMMAND"
CORRECTNESS_MODE=$CORRECTNESS_MODE
HAS_HIP=$HAS_HIP
HAS_TRITON=$HAS_TRITON
OUT
