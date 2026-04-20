# Action: Framework Rebuild

Rebuild a framework or operator library after code-level changes (dispatch fixes,
kernel source modifications, extension updates). Includes mandatory rollback plan
and verification that the correct code path is active after rebuild.

## Inputs
- Target library to rebuild (framework, operator library, or extension)
- Change description (what was modified and why)
- Rollback plan (how to undo if rebuild breaks things)

## Procedure

### Step 1: Pre-rebuild safety

```bash
# a) Identify what we're rebuilding
TARGET_LIB="$1"        # e.g. sglang, aiter, sgl-kernel, vllm
TARGET_DIR="$2"        # path to the library source
CHANGE_DESC="$3"       # human-readable change description

echo "=== Framework Rebuild: $TARGET_LIB ==="
echo "Change: $CHANGE_DESC"
echo "Directory: $TARGET_DIR"

# b) Create rollback snapshot
BACKUP_DIR="$RESULT_DIR/rollback/${TARGET_LIB}_$(date +%s)"
mkdir -p "$BACKUP_DIR"

# For git-managed repos:
cd "$TARGET_DIR"
if git rev-parse --git-dir > /dev/null 2>&1; then
    ROLLBACK_COMMIT=$(git rev-parse HEAD)
    git stash push -m "marathon-pre-rebuild-$(date +%s)"
    echo "Git rollback: git stash pop OR git checkout $ROLLBACK_COMMIT"
else
    # For non-git directories, copy modified files
    # (only copy what was changed, not entire lib)
    echo "Non-git directory — copying modified files to $BACKUP_DIR"
fi

# c) Record installed version before rebuild
python3 -c "
import importlib
try:
    mod = importlib.import_module('$TARGET_LIB'.replace('-', '_'))
    print(f'Before: {mod.__name__} v{getattr(mod, \"__version__\", \"unknown\")}')
    if hasattr(mod, '__file__'):
        print(f'Location: {mod.__file__}')
except ImportError:
    print(f'$TARGET_LIB not importable as Python module')
"
```

### Step 2: Rebuild

```bash
cd "$TARGET_DIR"

# Determine rebuild method
if [ -f "setup.py" ] || [ -f "pyproject.toml" ]; then
    echo "=== pip install -e . ==="
    pip install -e . --no-deps 2>&1 | tee "$RESULT_DIR/rebuild_${TARGET_LIB}.log"
    REBUILD_STATUS=$?
elif [ -f "CMakeLists.txt" ]; then
    echo "=== cmake build ==="
    mkdir -p build && cd build
    cmake .. -DCMAKE_BUILD_TYPE=Release 2>&1 | tee "$RESULT_DIR/rebuild_${TARGET_LIB}.log"
    make -j$(nproc) 2>&1 | tee -a "$RESULT_DIR/rebuild_${TARGET_LIB}.log"
    REBUILD_STATUS=$?
elif [ -f "Makefile" ]; then
    echo "=== make ==="
    make -j$(nproc) 2>&1 | tee "$RESULT_DIR/rebuild_${TARGET_LIB}.log"
    REBUILD_STATUS=$?
else
    echo "ERROR: No recognized build system in $TARGET_DIR"
    REBUILD_STATUS=1
fi

if [ $REBUILD_STATUS -ne 0 ]; then
    echo "REBUILD FAILED — initiating rollback"
    # Trigger Step 4 rollback
fi
```

### Step 3: Post-rebuild verification

**This step is MANDATORY.** A successful build does not mean the correct code path is active.

```bash
# a) Verify the library is importable
python3 -c "
import importlib
mod = importlib.import_module('$TARGET_LIB'.replace('-', '_'))
print(f'After: {mod.__name__} v{getattr(mod, \"__version__\", \"unknown\")}')
print(f'Location: {mod.__file__}')
"

# b) Verify the specific change is active
# This is change-specific. Examples:
#
# For a dispatch routing fix:
#   python3 -c "from framework.layer import dispatch_fn; import dis; dis.dis(dispatch_fn)"
#   → verify the correct branch is taken
#
# For a kernel source change:
#   python3 -c "import extension; print(extension.__file__)"
#   → verify the .so was actually rebuilt (check mtime)
#
# For an operator tuning config change:
#   python3 -c "from lib.configs import get_config; print(get_config(N=7168, K=4608))"
#   → verify new config is loaded

# c) Quick smoke test — does the server still launch?
echo "Running smoke test..."
# Start server, send 1 request, check output makes sense, kill server
# (use the same launch config from state)
```

### Step 4: Rollback (if needed)

```bash
cd "$TARGET_DIR"

echo "=== ROLLBACK: $TARGET_LIB ==="

if git rev-parse --git-dir > /dev/null 2>&1; then
    # Git rollback
    git stash pop 2>/dev/null || git checkout "$ROLLBACK_COMMIT"
    # Re-rebuild with original code
    pip install -e . --no-deps 2>&1
else
    # File-based rollback
    cp -r "$BACKUP_DIR"/* "$TARGET_DIR/"
    pip install -e . --no-deps 2>&1
fi

echo "Rollback complete. Verify with smoke test."
```

## Outputs
- Rebuilt library with changes applied
- Verification log confirming correct code path is active
- Rollback snapshot in `$RESULT_DIR/rollback/`
- Rebuild log in `$RESULT_DIR/rebuild_${TARGET_LIB}.log`

## Heuristic Update

- **Rebuild succeeded + correct path verified:** Boost framework-rebuild for other libraries.
  Log to `state.frameworks_rebuilt`.
- **Rebuild failed:** Rollback. Reduce score for this library's rebuild by 0.5×. If the
  build failure is a missing dependency, try installing it and retry (1 retry max).
- **Rebuild succeeded but wrong path still active:** This is a bug in our change.
  Do NOT retry blindly — investigate dispatch logic first via `deep-kernel-analysis.md`.

## Safety Notes

- ALWAYS create a rollback snapshot before rebuilding
- ALWAYS verify the change is active after rebuild (dispatch check)
- NEVER rebuild more than one library at a time — isolate changes for accurate measurement
- After rebuild, run a full E2E benchmark before claiming gain
