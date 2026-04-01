# Action: Kernel Integration

## Overview

Integrates a GEAK-optimized or manually-optimized kernel into the training stack
and benchmarks it as a normal optimization attempt.

## Inputs
- Optimized kernel source (from GEAK or manual optimization)
- Original kernel location (path in training stack)
- Current kept_overrides and kept_patches

## Procedure

### Step 1: Backup original

```bash
cp "$ORIGINAL_KERNEL_PATH" "${ORIGINAL_KERNEL_PATH}.bak"
```

### Step 2: Choose integration path

**Path A: Inductor cache patch** (if using torch.compile)
```python
# Replace kernel in Inductor cache
import shutil, os
shutil.copy2(inductor_file, inductor_file + ".bak")
# Patch the kernel function body
# Clear Triton binary cache to force recompilation
shutil.rmtree(os.path.expanduser("~/.triton/cache"), ignore_errors=True)
```

**Path B: Source file patch** (primary for distributed training)
```python
# Replace the kernel function in the source file
# Use AST-based patching for safety (not regex)
import ast

with open(ORIGINAL_KERNEL_PATH) as f:
    tree = ast.parse(f.read())

# Find and replace the kernel function
for node in ast.walk(tree):
    if isinstance(node, ast.FunctionDef) and node.name == KERNEL_NAME:
        # Replace function body with optimized version
        pass

with open(ORIGINAL_KERNEL_PATH, "w") as f:
    f.write(ast.unparse(tree))
```

**Path C: Monkey-patch at import** (least invasive)
```python
# Add a monkey-patch module that overrides the kernel at import time
import original_module
from optimized_kernel import optimized_fn
original_module.kernel_fn = optimized_fn
```

### Step 3: Full training benchmark

```bash
pkill -9 -f "primus/cli/main.py" 2>/dev/null; sleep 5
MASTER_PORT=$((MASTER_PORT + 1))

torchrun --nproc_per_node=$NUM_GPUS --master_port=$MASTER_PORT \
  -m primus.cli.main train pretrain \
  --config "$CONFIG_YAML" \
  $KEPT_OVERRIDES \
  profile=false use_pytorch_profiler=false \
  2>&1 | tee /tmp/attempt_kernel_N.log
```

### Step 4: KEEP or REVERT

- **If ms/iter improved:** KEEP the patch. Add to `kept_patches`.
- **If ms/iter same or worse:** REVERT: `cp "${ORIGINAL_KERNEL_PATH}.bak" "$ORIGINAL_KERNEL_PATH"`
- **If training crashes:** REVERT and log crash.

### Step 5: Post-integration re-profile (if kept)

If a kernel optimization is kept, push a `re-profile` action to discover
if the optimization exposed new bottlenecks.

## Outputs
- `actual_e2e_pct`: actual end-to-end speedup percentage
- KEEP/REVERT decision
- Backup files for revert

## Failure Handling
- If patch breaks import: revert from backup, log error
- If training hangs: kill after timeout, revert
- If numerical differences in loss: revert, kernel is not functionally correct
