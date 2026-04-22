# Action: Try Compiler / CMake Flag Overrides

## When to use
After env-vars have been exhausted (or in parallel for `hip-cmake-bench` /
`hpc-app` projects where flags often dominate).

## Catalog

### CMake / build-type
| Flag | Try | Notes |
|---|---|---|
| `-DCMAKE_BUILD_TYPE` | `Release`, `RelWithDebInfo` | Confirm Release |
| `-DCMAKE_HIP_ARCHITECTURES` | matches detected `$GPU_ARCH` | Critical — wrong arch silently runs slowly |
| `-DCMAKE_INTERPROCEDURAL_OPTIMIZATION` | `ON` | LTO; ~2-5% on dense workloads |
| `-DBUILD_SHARED_LIBS` | `OFF` | Static linking, faster dispatch |

### HIP / Clang flags (via `-DCMAKE_HIP_FLAGS=...`)
| Flag | Notes |
|---|---|
| `-O3` | Confirm |
| `-ffast-math` | Reorders FP; correctness gate catches breakage |
| `-mllvm -amdgpu-early-inline-all=true` | Aggressive inlining for kernel-heavy code |
| `-mllvm -amdgpu-function-calls=false` | Force-inline all device functions |
| `-mllvm -amdgpu-unroll-threshold=...` | Tune unroll factor (try 100, 200, 400) |
| `-fgpu-rdc` / `-fno-gpu-rdc` | RDC on/off; off is faster if no cross-TU device calls |
| `-Xclang -mllvm -Xclang -amdgpu-internalize-symbols` | Better IPO |

### CXX flags
| Flag | Notes |
|---|---|
| `-O3 -DNDEBUG` | Confirm |
| `-march=native` | Host-side — doesn't affect kernels but helps launch overhead |

## Procedure

### Step 1: Pick a single flag to add (or change)
```bash
EXTRA_HIP_FLAGS="-mllvm -amdgpu-early-inline-all=true"
ATTEMPT_DESCRIPTION="hip-flag: $EXTRA_HIP_FLAGS"
BUILD_DIR_SUFFIX="attempt-${ATTEMPT_ID}-inline-all"
```

### Step 2: Trigger build.md (clean build into isolated dir)
A clean build is mandatory — CMake caches old flags otherwise.

### Step 3: Trigger baseline.md (using the new $BUILD_DIR)
The bench command stays the same; only the binary changed.

### Step 4: Trigger correctness.md
Compile flags can break correctness (especially `-ffast-math`). Never skip.

### Step 5: Keep or revert
If KEEP, append to `$RESULT_DIR/kept_cmake.txt`:
```
CMAKE: -DCMAKE_BUILD_TYPE=Release
HIP:   -mllvm -amdgpu-early-inline-all=true
```
The next build action concatenates everything in this file.

## Combination Rule
Try flags one at a time. After all candidates tested, attempt the union of
winners as a final consolidation pass.

## Outputs
- `$RESULT_DIR/kept_cmake.txt` (sourced by `build.md` for subsequent builds)
- Entry in `results.tsv`
