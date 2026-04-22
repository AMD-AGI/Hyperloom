# Action: Build the Project

## Inputs
- `$REPO_ROOT`, `$RESULT_DIR`
- `$BUILD_COMMAND` (from detected.env)
- `$EXTRA_CMAKE_FLAGS` / `$EXTRA_CXX_FLAGS` / `$EXTRA_HIP_FLAGS` (optional, from compile-flags action)
- `$BUILD_DIR_SUFFIX` (optional, e.g. "attempt-3" — isolates builds when changing flags)

## Procedure

### Step 1: Decide build directory
For CMake projects, isolate builds when compile flags change:
```bash
if [ -n "${EXTRA_CMAKE_FLAGS:-}${EXTRA_CXX_FLAGS:-}${EXTRA_HIP_FLAGS:-}" ]; then
    BUILD_DIR="$REPO_ROOT/build-${BUILD_DIR_SUFFIX:-baseline}"
else
    BUILD_DIR="${BUILD_DIR:-$REPO_ROOT/build}"
fi
mkdir -p "$BUILD_DIR"
```

### Step 2: Run the build
```bash
cd "$REPO_ROOT"

case "$PROJECT_CLASS" in
  hip-cmake-bench|hpc-app)
      # Allow detect.md to override the default cmake invocation
      if [ -n "${EXTRA_CMAKE_FLAGS:-}" ]; then
          cmake -S . -B "$BUILD_DIR" $EXTRA_CMAKE_FLAGS \
              -DCMAKE_CXX_FLAGS="${EXTRA_CXX_FLAGS:-}" \
              -DCMAKE_HIP_FLAGS="${EXTRA_HIP_FLAGS:-}"
          cmake --build "$BUILD_DIR" -j"$(nproc)" 2>&1 | tee "$RESULT_DIR/build.log"
      else
          eval "$BUILD_COMMAND" 2>&1 | tee "$RESULT_DIR/build.log"
      fi
      ;;
  pytorch-script|triton-collection)
      eval "$BUILD_COMMAND" 2>&1 | tee "$RESULT_DIR/build.log"
      ;;
  *)
      eval "$BUILD_COMMAND" 2>&1 | tee "$RESULT_DIR/build.log"
      ;;
esac
```

### Step 3: Capture build success
```bash
BUILD_EXIT=${PIPESTATUS[0]}
if [ $BUILD_EXIT -ne 0 ]; then
    echo "BUILD_FAILED" > "$RESULT_DIR/build_status.txt"
    echo "ERROR: build failed with exit $BUILD_EXIT — see $RESULT_DIR/build.log"
    exit $BUILD_EXIT
fi
echo "BUILD_OK" > "$RESULT_DIR/build_status.txt"
echo "BUILD_DIR=$BUILD_DIR" >> "$RESULT_DIR/state.env"
```

## Outputs
- `$BUILD_DIR/` populated
- `$RESULT_DIR/build.log` for diagnosis
- `BUILD_DIR` exported for downstream actions

## Failure Handling
- Build failure: REVERT the most recent change, mark attempt as INVALID,
  decrement consecutive_discards (don't penalize the agent for a transient build
  flake — but record it).
- For 3 build failures in a row: stop and surface to the user.
