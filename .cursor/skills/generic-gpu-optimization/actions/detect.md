# Action: Auto-Detect Project

## Goal
Determine `project_class`, `build_command`, `bench_command`, `test_command`,
`kernel_langs`, and `correctness_mode`. Write everything to
`$RESULT_DIR/detected.env` so subsequent actions can dispatch.

## Inputs
- `$REPO_ROOT`, `$RESULT_DIR`
- Optional user overrides: `$BUILD_COMMAND`, `$BENCH_COMMAND`, `$TEST_COMMAND`,
  `$BENCH_METRIC_REGEX`

## Procedure

### Step 1: Run the heuristic detector
```bash
"$SKILL_ROOT/scripts/detect_project.sh" "$REPO_ROOT" > "$RESULT_DIR/detected.env"
cat "$RESULT_DIR/detected.env"
source "$RESULT_DIR/detected.env"
```

The detector classifies into one of:
- `hip-cmake-bench` — `CMakeLists.txt` + a `bench/` or `benchmarks/` dir with
  Google Benchmark (`benchmark::benchmark` link), Catch2, or NVBench
- `pytorch-script` — `pyproject.toml` or `setup.py` + at least one `.py` that
  imports `torch` AND has a `__main__` block with timing output
- `triton-collection` — Python code that imports `triton` and has a `bench.py`,
  `benchmark.py`, or `benchmarks/` directory
- `hpc-app` — `Makefile` or `CMakeLists.txt` with `--bench` / `-b` flag in main
- `unknown` — fallback

### Step 2: Locate kernel sources
```bash
KERNEL_LANGS=""
[ -n "$(find "$REPO_ROOT" -name '*.hip' -o -name '*.cuh' -o -name '*.cu' 2>/dev/null | head -1)" ] && KERNEL_LANGS="$KERNEL_LANGS hip"
grep -r -l --include='*.py' '@triton.jit' "$REPO_ROOT" 2>/dev/null | head -1 >/dev/null && KERNEL_LANGS="$KERNEL_LANGS triton"
grep -r -l --include='*.py' 'torch.compile\|torch.compile(' "$REPO_ROOT" 2>/dev/null | head -1 >/dev/null && KERNEL_LANGS="$KERNEL_LANGS torch-compile"
echo "KERNEL_LANGS=\"$(echo $KERNEL_LANGS | xargs)\"" >> "$RESULT_DIR/detected.env"
```

### Step 3: Determine correctness mode
Priority order:
1. If `ctest` is configured (CMake build dir contains `CTestTestfile.cmake`) → `tests`, `TEST_COMMAND="ctest --output-on-failure"`
2. If `pytest` config or `tests/` dir with `test_*.py` → `tests`, `TEST_COMMAND="pytest -x"`
3. If a known benchmark binary exists, capture its FIRST baseline output as a golden file → `golden-output`
4. Else → `none` (warn the user)

Append to `detected.env`:
```bash
echo "CORRECTNESS_MODE=$CORRECTNESS_MODE"   >> "$RESULT_DIR/detected.env"
echo "TEST_COMMAND=\"$TEST_COMMAND\""        >> "$RESULT_DIR/detected.env"
```

### Step 4: Apply user overrides
Anything passed in via `$BUILD_COMMAND`, `$BENCH_COMMAND`, `$TEST_COMMAND`, or
`$BENCH_METRIC_REGEX` REPLACES the detected value:

```bash
[ -n "${BUILD_COMMAND:-}" ] && sed -i "s|^BUILD_COMMAND=.*|BUILD_COMMAND=\"$BUILD_COMMAND\"|" "$RESULT_DIR/detected.env"
[ -n "${BENCH_COMMAND:-}" ] && sed -i "s|^BENCH_COMMAND=.*|BENCH_COMMAND=\"$BENCH_COMMAND\"|" "$RESULT_DIR/detected.env"
[ -n "${TEST_COMMAND:-}"  ] && sed -i "s|^TEST_COMMAND=.*|TEST_COMMAND=\"$TEST_COMMAND\"|"   "$RESULT_DIR/detected.env"
```

### Step 5: Sanity print
```bash
echo "=== Detection Summary ==="
echo "  project_class:    $PROJECT_CLASS"
echo "  build_command:    $BUILD_COMMAND"
echo "  bench_command:    $BENCH_COMMAND"
echo "  bench_metric:     $BENCH_METRIC ($BENCH_METRIC_REGEX)"
echo "  test_command:     $TEST_COMMAND"
echo "  correctness_mode: $CORRECTNESS_MODE"
echo "  kernel_langs:     $KERNEL_LANGS"
```

## Outputs
- `$RESULT_DIR/detected.env` — sourceable, contains everything subsequent actions
  need to know about the project shape

## Failure Handling
- If detector returns `unknown` AND no overrides supplied: ASK the user for
  `BUILD_COMMAND`, `BENCH_COMMAND`, `TEST_COMMAND` and re-run this action.
- If `BENCH_COMMAND` is empty after detection + overrides: STOP with a clear error.
