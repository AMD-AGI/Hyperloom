# hipBLASLt Benchmarking, Build, and Testing Reference

Source: `rocm-libraries/projects/hipblaslt/` in the rocm-libraries monorepo.

---

## 1. Building from Source

### 1.1 Quick Build with install.sh

```bash
cd rocm-libraries/projects/hipblaslt

# Library only (first time, installs dependencies)
./install.sh -d

# Library + clients (bench + test + samples)
./install.sh -dc

# Library + clients + install to /opt/rocm
./install.sh -idc

# Target specific GPU architecture
./install.sh -dc -a gfx942          # MI300X
./install.sh -dc -a gfx950          # MI350X
./install.sh -dc -a "gfx942;gfx950" # multiple targets

# Debug build
./install.sh -dc -g

# RelWithDebInfo (for profiling)
./install.sh -dc -k

# Client-only rebuild (skip Tensile library generation)
./install.sh -c -a gfx942 -n

# Static library
./install.sh --static
```

Key install.sh flags:
| Flag | Description |
|------|-------------|
| `-d` | Install build dependencies (only needed once) |
| `-c` | Build clients (hipblaslt-bench, hipblaslt-test, samples) |
| `-i` | Install to /opt/rocm (requires sudo) |
| `-a <arch>` | GPU architecture target(s), e.g., `gfx942`, `gfx950`, `all` |
| `-g` | Debug build |
| `-k` | RelWithDebInfo build |
| `-n` / `--client-only` | Skip Tensile library build, use prebuilt |
| `--static` | Build static library |
| `--skip_rocroller` | Skip rocRoller backend |
| `--no-compress` | Don't compress assembly code objects |
| `--logic-yaml-filter` | Filter logic files, e.g., `gfx942/Equality/*` |

### 1.2 Manual CMake Build

```bash
mkdir -p build/release && cd build/release

CXX=/opt/rocm/bin/amdclang++ cmake \
    -DCMAKE_BUILD_TYPE=Release \
    -DGPU_TARGETS="gfx950" \
    -DHIPBLASLT_ENABLE_FETCH=ON \
    -DHIPBLASLT_ENABLE_CLIENT=ON \
    -DHIPBLASLT_BUILD_TESTING=ON \
    -DHIPBLASLT_ENABLE_SAMPLES=ON \
    ../..

make -j$(nproc)
```

### 1.3 Important CMake Options

| Option | Default | Description |
|--------|---------|-------------|
| `GPU_TARGETS` | `""` (all) | Semicolon-separated list: `gfx942;gfx950` |
| `HIPBLASLT_ENABLE_DEVICE` | ON | Build Tensile device libraries |
| `HIPBLASLT_ENABLE_CLIENT` | ON | Build bench/test/samples |
| `HIPBLASLT_BUILD_TESTING` | ON | Build hipblaslt-test |
| `HIPBLASLT_ENABLE_SAMPLES` | ON | Build sample programs |
| `HIPBLASLT_ENABLE_LAZY_LOAD` | ON | Lazy load code objects (reduces RAM) |
| `HIPBLASLT_ENABLE_ROCROLLER` | ON | Enable rocRoller JIT backend |
| `HIPBLASLT_ENABLE_MARKER` | ON | Enable ROCProfiler markers |
| `HIPBLASLT_BUILD_SHARED_LIBS` | ON | Shared vs static library |
| `HIPBLASLT_ENABLE_ASAN` | OFF | Address sanitizer |
| `HIPBLASLT_ENABLE_COVERAGE` | OFF | Code coverage (Debug/RelWithDebInfo only) |
| `CMAKE_INSTALL_PREFIX` | `/opt/rocm` | Install path |

### 1.4 Dependencies

- ROCm (hip, hipblas-common)
- msgpack-cxx (for library data parsing)
- LAPACK + gfortran (for clients)
- GoogleTest (for tests, built from deps/)
- BLIS (optional CPU reference for testing)
- amd_smi (for efficiency monitoring, Linux only)

### 1.5 Build Outputs

After build, executables are in `build/release/clients/`:
- `hipblaslt-bench` -- performance benchmarking
- `hipblaslt-test` -- correctness testing (GoogleTest)
- `hipblaslt-bench-groupedgemm-fixed-mk` -- grouped GEMM bench
- `hipblaslt-bench-extop-layernorm` -- layernorm ext-op bench
- `hipblaslt-bench-extop-softmax` -- softmax ext-op bench
- `hipblaslt-bench-extop-amax` -- amax ext-op bench
- `sample_hipblaslt_*` -- various API usage samples

---

## 2. hipblaslt-bench Command-Line Reference

### 2.1 Matrix Dimensions and Layout

| Flag | Default | Description |
|------|---------|-------------|
| `-m` / `--sizem` | 128 | M dimension (rows of op(A) and C/D) |
| `-n` / `--sizen` | 128 | N dimension (columns of op(B) and C/D) |
| `-k` / `--sizek` | 128 | K dimension (columns of op(A), rows of op(B)) |
| `--lda` | auto | Leading dimension of A |
| `--ldb` | auto | Leading dimension of B |
| `--ldc` | auto | Leading dimension of C |
| `--ldd` | auto | Leading dimension of D |
| `--lde` | auto | Leading dimension of E (auxiliary) |
| `--transA` | N | Transpose A: N or T |
| `--transB` | N | Transpose B: N or T |
| `--batch_count` | 1 | Batch count for batched GEMM |
| `--stride_a/b/c/d/e` | auto | Strides for strided batched |
| `--any_stride` | false | Don't auto-adjust strides |

For grouped GEMM, pass multiple values: `-m 128 -m 256 -n 512 -n 1024 -k 64 -k 128 --grouped_gemm`.

### 2.2 Data Types

| Flag | Default | Description |
|------|---------|-------------|
| `-r` / `--precision` | `f16_r` | Sets A,B,C,D type uniformly |
| `--a_type` | (from -r) | Override A precision |
| `--b_type` | (from -r) | Override B precision |
| `--c_type` | (from -r) | Override C precision |
| `--d_type` | (from -r) | Override D precision |
| `--compute_type` | `f32_r` | Compute type |
| `--compute_input_typeA` | (none) | Compute input cast for A |
| `--compute_input_typeB` | (none) | Compute input cast for B |
| `--scale_type` | (auto) | Scalar alpha/beta type |
| `--bias_type` | (same as D) | Bias vector type |
| `--aux_type` | (same as D) | Auxiliary matrix E type |

**Precision string values:**
- Float: `f32_r`, `f64_r`
- Half: `f16_r`, `bf16_r`
- Integer: `i8_r`, `i32_r`
- FP8 (OCP, gfx950/gfx12): `f8_r`, `bf8_r`
- FP8 (FNUZ, gfx942): `f8_fnuz_r`, `bf8_fnuz_r`
- Sub-byte (gfx950): `f4_r` (FP4 E2M1), `f6e2m3_r`, `f6e3m2_r`

**Compute type values:** `f32_r`, `f16_r`, `f64_r`, `i32_r`, `xf32_r` (TF32), `f32_bf16_r`

NOTE: On gfx942, OCP FP8 types (`f8_r`, `bf8_r`) are automatically converted to FNUZ equivalents.

### 2.3 Scaling

| Flag | Default | Description |
|------|---------|-------------|
| `--scaleA` | 0 | Scale format for A: 0=none, 1=scalar, 2=vector, 3=Block_32_UE8M0, 4=Block_16_UE8M0, 5=Block_32_UE4M3, 6=Block_16_UE4M3, 7=Block_32_UE5M3, 8=Block_16_UE5M3, 1001=block_preswizzled_32x8 |
| `--scaleB` | 0 | Scale format for B (same options) |
| `--scaleC` | 0 | Scale for C: 0=none, 1=scalar |
| `--scaleD` | 0 | Scale for D: 0=none, 1=scalar |
| `--scaleAlpha_vector` | false | Apply per-column alpha scaling |
| `--amaxScaleA` | false | Scale A by abs-max of A |
| `--amaxScaleB` | false | Scale B by abs-max of B |
| `--amaxD` | false | Output amax of D |
| `--alpha` | 1.0 | Scalar alpha |
| `--beta` | 0.0 | Scalar beta |

Block scaling requires FP8/FP6/FP4 input types and f32 compute type. C/D must be f32/f16/bf16.

### 2.4 Epilogue / Activation / Bias

| Flag | Default | Description |
|------|---------|-------------|
| `--activation_type` | `none` | Epilogue: `none`, `gelu`, `relu`, `swish`, `clamp` |
| `--activation_arg1` | 0 | Activation parameter 1 (e.g., clamp min) |
| `--activation_arg2` | inf | Activation parameter 2 (e.g., clamp max) |
| `--bias_vector` | false | Enable bias vector |
| `--bias_source` | `d` | Bias source: `a`, `b`, or `d` |
| `--use_e` | false | Enable auxiliary output (for GELU aux, gradient input) |
| `--gradient` | false | Enable gradient mode (for backward pass epilogues) |

### 2.5 Algorithm Selection

| Flag | Default | Description |
|------|---------|-------------|
| `--algo_method` | `heuristic` | `heuristic`, `all`, or `index` |
| `--requested_solution` | 1 | Number of solutions to try. -1 = all solutions |
| `--solution_index` | -1 | Specific solution index (with `--algo_method index`) |
| `--print_kernel_info` | false | Print solution name, kernel name, and index |
| `--skip_slow_solution_ratio` | 0.0 | Skip solutions slower than best by this ratio (0-1) |

### 2.6 Benchmarking Control

| Flag | Default | Description |
|------|---------|-------------|
| `-i` / `--iters` | 10 | Timing loop iterations |
| `-j` / `--cold_iters` | 2 | Warm-up iterations before timing |
| `--rotating` | 0 | Rotating buffer size in MB (avoids cache effects) |
| `--flush` | false | Flush instruction cache between iterations |
| `--use_gpu_timer` | false | Use hipEventElapsedTime instead of host timer |
| `--workspace` | 128MB | Fixed workspace size in bytes |
| `--device` | 0 | GPU device ID |
| `-v` / `--verify` | false | Validate against CPU reference |
| `--api_method` | `c` | API method: `c`, `mix`, `cpp` |
| `--initialization` | `hpl` | Matrix init: `rand_int`, `trig_float`, `hpl`, `zero`, `norm_dist`, `uniform_01` |

### 2.7 Tensor Swizzling

| Flag | Default | Description |
|------|---------|-------------|
| `--swizzleA` | false | Enable A swizzle (requires TN layout, FP16/BF16/FP8) |
| `--swizzleB` | false | Enable B swizzle (requires TN layout, FP16/BF16/FP8) |

### 2.8 Tuning Parameters

| Flag | Default | Description |
|------|---------|-------------|
| `--splitk` | (solution default) | Split-K value (requires `--api_method mix` or `cpp`) |
| `--wgm` | (solution default) | Workgroup mapping (requires `--api_method mix` or `cpp`) |

---

## 3. Benchmark Output Format

Output is CSV with a header line and value line per solution. The format is:

```
[<index>]:<header_fields>
    <value_fields>
```

### 3.1 Standard Columns

```
transA,transB,grouped_gemm,batch_count,m,n,k,alpha,lda,stride_a,beta,ldb,stride_b,
ldc,stride_c,ldd,stride_d,a_type,b_type,c_type,d_type,compute_type,
scaleA,scaleB,scaleC,scaleD,amaxD,activation_type,bias_vector,bias_type,
rotating_buffer,flush,use_gpu_timer,hipblaslt-Gflops,hipblaslt-GB/s,us
```

### 3.2 Key Performance Metrics

| Column | Unit | Description |
|--------|------|-------------|
| `hipblaslt-Gflops` | GFLOPS | Achieved throughput (2*M*N*K*batch / time) |
| `hipblaslt-GB/s` | GB/s | Memory bandwidth |
| `us` | microseconds | Average kernel time per iteration |
| `solution_index` | int | Internal solution/kernel index |

### 3.3 With --print_kernel_info

When `--print_kernel_info` is set, additional lines appear after the CSV:
```
    --Solution index: 56537
    --Solution name:  <tensile_solution_name>
    --kernel name:    <HSA_kernel_name>
```

### 3.4 With Verification (-v)

Additional columns: `CPU-Gflops`, `CPU-us`, `norm_error`, `atol`, `rtol`

### 3.5 Frequency Monitoring (Environment Variables)

| Env Var | Extra Columns |
|---------|---------------|
| `HIPBLASLT_BENCH_FREQ=1` | `lowest_avg_freq`, `lowest_median_freq`, `avg_MCLK`, `median_MCLK` |
| `HIPBLASLT_BENCH_FREQ_ALL=1` | Per-XCD frequencies: `avg_freq0..7`, `median_freq0..7`, `avg_MCLK`, `median_MCLK` |
| `HIPBLASLT_BENCH_PERF=1` | `num_cu`, `tiles_per_cu`, `tile0_gran`, `tile1_gran`, `cu_gran`, `wave_gran`, `total_gran`, `mem_read_bytes`, `mem_write_bytes`, `efficiency` |

---

## 4. Environment Variables

### 4.1 Logging and Debugging

| Variable | Values | Description |
|----------|--------|-------------|
| `HIPBLASLT_LOG_LEVEL` | 0-5 | 0=off, 1=error, 2=trace, 3=hints, 4=info, 5=API trace |
| `HIPBLASLT_LOG_MASK` | bitmask | 1=error, 2=trace, 4=hints, 8=info, 16=API trace, **32=bench**, 64=profile, 128=extended profile |
| `HIPBLASLT_LOG_FILE` | path | Log file (%i = PID); stdout if unset |
| `HIPBLASLT_ENABLE_MARKER` | 0/1 | Enable ROCProfiler marker trace |

**Critical:** `HIPBLASLT_LOG_MASK=32` generates `hipblaslt-bench` command lines from any hipBLASLt API call. This is the standard way to extract reproducible bench commands from an application.

### 4.2 Offline Tuning

| Variable | Description |
|----------|-------------|
| `HIPBLASLT_TUNING_FILE` | Path to write tuning results (best solution indices). Changes bench defaults: iters=1000, cold_iters=1000, rotating=512, requested_solution=-1 |
| `HIPBLASLT_TUNING_OVERRIDE_FILE` | Path to load tuning results and override kernel selection |
| `HIPBLASLT_TUNING_USER_MAX_WORKSPACE` | Max workspace bytes during tuning (default: 128MB) |

### 4.3 Stream-K / Origami

| Variable | Description |
|----------|-------------|
| `TENSILE_SOLUTION_SELECTION_METHOD` | 0=standard (default), 2=Origami+Stream-K. No effect on MI350 (Stream-K always used) |
| `TENSILE_STREAMK_DYNAMIC_GRID` | 0=disable, 1-5=various strategies, 6=auto (default) |
| `TENSILE_STREAMK_FIXED_GRID` | Override grid size (number of workgroups) |
| `TENSILE_STREAMK_MAX_CUS` | Limit max compute units for Stream-K |

### 4.4 Library Paths and Debug

| Variable | Description |
|----------|-------------|
| `HIPBLASLT_TENSILE_LIBPATH` | Path to Tensile library directory containing `TensileLibrary_lazy_<arch>.dat` and `.co` files |
| `TENSILE_DB` | Debug bitmask for Tensile. `0x2|0x4` enables solution tag logging |
| `HIPBLASLT_OVERRIDE_COMPUTE_TYPE_XF32` | -1=off, 0=F32, 1=XF32/TF32, 2=F32_BF16 |

---

## 5. hipblaslt-test (Correctness Verification)

GoogleTest-based regression suite:

```bash
# Full test suite
./hipblaslt-test

# Quick smoke tests
./hipblaslt-test --gtest_filter=*quick*

# DRelu gradient tests
./hipblaslt-test --gtest_filter=*drelu*

# Specific precision tests
./hipblaslt-test --gtest_filter=*f16*

# List available tests
./hipblaslt-test --gtest_list_tests

# With prebuilt libraries (skip device library build)
HIPBLASLT_TENSILE_LIBPATH=/path/to/Tensile/library/ ./hipblaslt-test
```

Precompile database for faster repeated runs:
```bash
./hipblaslt-test --precompile=hipblaslt-test-precompile.db
```

---

## 6. Practical Benchmark Recipes

### 6.1 Basic FP16 GEMM (default)

```bash
./hipblaslt-bench -m 4096 -n 4096 -k 4096
```

### 6.2 FP8 GEMM for LLM Inference (gfx950)

```bash
# Single token decode shape
./hipblaslt-bench -m 1 -n 4096 -k 4096 \
    --a_type f8_r --b_type f8_r --c_type f16_r --d_type f16_r \
    --compute_type f32_r --scaleA 1 --scaleB 1 \
    --transA N --transB N -i 100 --print_kernel_info

# Prefill shape
./hipblaslt-bench -m 2048 -n 4096 -k 4096 \
    --a_type f8_r --b_type f8_r --c_type f16_r --d_type f16_r \
    --compute_type f32_r --scaleA 1 --scaleB 1 \
    --transA N --transB N -i 100 --print_kernel_info
```

### 6.3 FP8 GEMM for LLM Inference (gfx942 / MI300X)

```bash
# Auto-converts f8_r -> f8_fnuz_r on gfx942
./hipblaslt-bench -m 1 -n 4096 -k 4096 \
    --a_type f8_r --b_type f8_r --c_type f16_r --d_type f16_r \
    --compute_type f32_r --scaleA 1 --scaleB 1 -i 100
```

### 6.4 BF16 GEMM with GELU Epilogue (Training Forward)

```bash
./hipblaslt-bench -m 4096 -n 11008 -k 4096 \
    --precision bf16_r --compute_type f32_r \
    --activation_type gelu --use_e \
    --transA N --transB N -i 100 --print_kernel_info
```

### 6.5 BF16 GEMM with Bias (LLM Linear Layer)

```bash
./hipblaslt-bench -m 4096 -n 4096 -k 4096 \
    --precision bf16_r --compute_type f32_r \
    --bias_vector --bias_type bf16_r \
    -i 100 --print_kernel_info
```

### 6.6 FP8 with Block Scaling (MXFP)

```bash
./hipblaslt-bench -m 4096 -n 4096 -k 4096 \
    --a_type f8_r --b_type f8_r --c_type bf16_r --d_type bf16_r \
    --compute_type f32_r \
    --scaleA 3 --scaleB 3 \
    -i 100 --print_kernel_info
```

Scale format codes: 3=Block_32_UE8M0 (E8M0 scales per 32 elements), 4=Block_16_UE8M0, 5=Block_32_UE4M3, etc.

### 6.7 TN Layout (Common in LLM Weight-Stationary)

```bash
./hipblaslt-bench -m 4096 -n 4096 -k 4096 \
    --precision bf16_r --transA T --transB N \
    -i 100 --print_kernel_info
```

### 6.8 Batched GEMM (Multi-Head Attention QK^T)

```bash
./hipblaslt-bench -m 128 -n 128 -k 128 \
    --precision f16_r --batch_count 32 \
    --transA N --transB T -i 100
```

### 6.9 Grouped GEMM (MoE)

```bash
./hipblaslt-bench \
    -m 256 -m 512 -m 128 \
    -n 4096 -n 4096 -n 4096 \
    -k 4096 -k 4096 -k 4096 \
    --grouped_gemm --precision bf16_r \
    --api_method cpp -i 100 --print_kernel_info
```

### 6.10 Find the Best Solution (Exhaustive Search)

```bash
./hipblaslt-bench -m 4096 -n 4096 -k 4096 \
    --precision bf16_r --algo_method all \
    --skip_slow_solution_ratio 0.5 \
    -i 50 --print_kernel_info
```

### 6.11 Benchmark a Specific Solution Index

```bash
./hipblaslt-bench -m 4096 -n 4096 -k 4096 \
    --precision bf16_r --algo_method index \
    --solution_index 56537 -i 100 --print_kernel_info
```

### 6.12 Cache-Bypassing Measurement (Rotating Buffer)

```bash
./hipblaslt-bench -m 4096 -n 4096 -k 4096 \
    --precision bf16_r --rotating 512 \
    -i 100 --cold_iters 20
```

512 MB rotating buffer exceeds L2 cache, measuring true HBM bandwidth.

### 6.13 With GPU Timer (More Accurate for Small Kernels)

```bash
./hipblaslt-bench -m 16 -n 16 -k 4096 \
    --precision bf16_r --use_gpu_timer -i 1000
```

### 6.14 Efficiency Analysis

```bash
HIPBLASLT_BENCH_PERF=1 ./hipblaslt-bench -m 4096 -n 4864 -k 32896 \
    --precision bf16_r --use_gpu_timer -i 416 --cold_iters 416
```

Reports: `num_cu`, `tiles_per_cu`, granularity metrics, `mem_read_bytes`, `mem_write_bytes`, `efficiency`.

### 6.15 Stream-K Comparison

```bash
# Default (data-parallel)
./hipblaslt-bench -m 4096 -n 4096 -k 4096 --precision bf16_r -i 100

# Stream-K
TENSILE_SOLUTION_SELECTION_METHOD=2 \
./hipblaslt-bench -m 4096 -n 4096 -k 4096 --precision bf16_r -i 100
```

### 6.16 Mixed Precision (FP8 input, FP16 compute)

```bash
./hipblaslt-bench -m 1024 -n 512 -k 1024 \
    --a_type f8_r --b_type f16_r --c_type f32_r --d_type f32_r \
    --compute_type f32_r --scaleA 1 -i 100
```

### 6.17 Correctness Verification of a Specific Shape

```bash
./hipblaslt-bench -m 1024 -n 512 -k 1024 --precision f16_r -v
```

### 6.18 Split-K Tuning

```bash
./hipblaslt-bench -m 256 -n 256 -k 65536 \
    --precision bf16_r --api_method cpp \
    --splitk 4 -i 100 --print_kernel_info
```

### 6.19 Workgroup Mapping Tuning

```bash
./hipblaslt-bench -m 4096 -n 4096 -k 4096 \
    --precision bf16_r --api_method cpp \
    --wgm 8 -i 100 --print_kernel_info
```

---

## 7. Offline Tuning Workflow

### 7.1 Generate Bench Commands from Application

```bash
# Step 1: Run your application with bench logging enabled
export HIPBLASLT_LOG_MASK=32
python my_llm_inference.py
# Output: hipblaslt-bench --api_method c -m 1024 -n 512 -k 1024 ...
```

### 7.2 Tune and Save Results

```bash
# Step 2: Run exhaustive tuning
export HIPBLASLT_TUNING_FILE=tuning.txt
./hipblaslt-bench --api_method c -m 1024 -n 512 -k 1024 \
    --a_type f16_r --b_type f16_r --c_type f16_r --d_type f16_r \
    --compute_type f32_r
# Defaults change to: iters=1000, cold_iters=1000, rotating=512, all solutions
```

### 7.3 Apply Tuning Results at Runtime

```bash
# Step 3: Use tuned kernels in your application
export HIPBLASLT_TUNING_OVERRIDE_FILE=tuning.txt
python my_llm_inference.py
```

### 7.4 Verify Tuning Was Applied

```bash
export HIPBLASLT_TUNING_OVERRIDE_FILE=tuning.txt
./hipblaslt-bench --api_method c -m 1024 -n 512 -k 1024 \
    --a_type f16_r --b_type f16_r --c_type f16_r --d_type f16_r \
    --compute_type f32_r --algo_method heuristic \
    --requested_solution 1 --print_kernel_info
# Should show the tuned solution_index
```

NOTE: Tuning indices are NOT portable across library versions or GPU architectures.

---

## 8. Sample Code Patterns

### 8.1 Basic GEMM (C API)

```cpp
#include <hipblaslt/hipblaslt.h>

// 1. Create handle
hipblasLtHandle_t handle;
hipblasLtCreate(&handle);

// 2. Create matrix layouts
hipblasLtMatrixLayout_t matA, matB, matC, matD;
hipblasLtMatrixLayoutCreate(&matA, HIP_R_16F, m, k, m);
hipblasLtMatrixLayoutCreate(&matB, HIP_R_16F, k, n, k);
hipblasLtMatrixLayoutCreate(&matC, HIP_R_16F, m, n, m);
hipblasLtMatrixLayoutCreate(&matD, HIP_R_16F, m, n, m);

// 3. Create matmul descriptor
hipblasLtMatmulDesc_t matmul;
hipblasLtMatmulDescCreate(&matmul, HIPBLAS_COMPUTE_32F, HIP_R_32F);
hipblasLtMatmulDescSetAttribute(matmul, HIPBLASLT_MATMUL_DESC_TRANSA, &trans_a, sizeof(int32_t));
hipblasLtMatmulDescSetAttribute(matmul, HIPBLASLT_MATMUL_DESC_TRANSB, &trans_b, sizeof(int32_t));

// 4. Set epilogue
hipblasLtEpilogue_t epilogue = HIPBLASLT_EPILOGUE_DEFAULT;
hipblasLtMatmulDescSetAttribute(matmul, HIPBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue));

// 5. Get heuristic
hipblasLtMatmulPreference_t pref;
hipblasLtMatmulPreferenceCreate(&pref);
hipblasLtMatmulPreferenceSetAttribute(pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                       &max_workspace_size, sizeof(max_workspace_size));
hipblasLtMatmulHeuristicResult_t heuristicResult[1];
int returnedAlgoCount = 0;
hipblasLtMatmulAlgoGetHeuristic(handle, matmul, matA, matB, matC, matD,
                                 pref, 1, heuristicResult, &returnedAlgoCount);

// 6. Run GEMM
hipblasLtMatmul(handle, matmul, &alpha, d_a, matA, d_b, matB, &beta,
                 d_c, matC, d_d, matD, &heuristicResult[0].algo,
                 d_workspace, workspace_size, stream);

// 7. Cleanup
hipblasLtMatrixLayoutDestroy(matA); // ... etc
hipblasLtMatmulDescDestroy(matmul);
hipblasLtMatmulPreferenceDestroy(pref);
hipblasLtDestroy(handle);
```

### 8.2 Extension API (C++ style, recommended)

```cpp
#include <hipblaslt/hipblaslt-ext.hpp>

hipblaslt_ext::GemmPreference gemmPref;
gemmPref.setMaxWorkspaceBytes(max_workspace_size);

hipblaslt_ext::Gemm gemm(handle, trans_a, trans_b,
    HIP_R_16F, HIP_R_16F, HIP_R_16F, HIP_R_16F, HIPBLAS_COMPUTE_32F);

hipblaslt_ext::GemmEpilogue epilogue;  // default = HIPBLASLT_EPILOGUE_DEFAULT
hipblaslt_ext::GemmInputs inputs;
inputs.setA(d_a); inputs.setB(d_b); inputs.setC(d_c); inputs.setD(d_d);
inputs.setAlpha(&alpha); inputs.setBeta(&beta);
gemm.setProblem(m, n, k, batch_count, epilogue, inputs);

std::vector<hipblasLtMatmulHeuristicResult_t> heuristicResult;
gemm.algoGetHeuristic(1, gemmPref, heuristicResult);

gemm.setMaxWorkspaceBytes(max_workspace_size);
gemm.initialize(heuristicResult[0].algo, d_workspace);
gemm.run(stream);
```

### 8.3 Get All Algorithms (Extension API)

```cpp
std::vector<hipblasLtMatmulHeuristicResult_t> heuristicResult;
hipblaslt_ext::getAllAlgos(handle, hipblaslt_ext::GemmType::HIPBLASLT_GEMM,
    trans_a, trans_b, HIP_R_16F, HIP_R_16F, HIP_R_16F, HIP_R_16F,
    HIPBLAS_COMPUTE_32F, heuristicResult);

// Filter valid algorithms
for (size_t i = 0; i < heuristicResult.size(); i++) {
    size_t workspaceSizeInBytes = 0;
    if (gemm.isAlgoSupported(heuristicResult[i].algo, workspaceSizeInBytes)
        == HIPBLAS_STATUS_SUCCESS) {
        if (workspaceSizeInBytes <= max_workspace_size) {
            // This algorithm is valid
        }
    }
}
```

### 8.4 Mixed Precision with ScaleA (Extension API)

```cpp
hipblaslt_ext::Gemm gemm(handle, trans_a, trans_b,
    HIP_R_8F_E4M3_FNUZ, HIP_R_16F, HIP_R_32F, HIP_R_32F,
    HIPBLAS_COMPUTE_32F_FAST_16F);

hipblaslt_ext::GemmInputs inputs;
inputs.setA(d_a); inputs.setB(d_b); inputs.setC(d_c); inputs.setD(d_d);
inputs.setAlpha(&alpha); inputs.setBeta(&beta);
inputs.setScaleA(d_scaleA);  // device pointer
gemm.setProblem(m, n, k, batch_count, epilogue, inputs);
```

### 8.5 Batched GEMM Layout Configuration

```cpp
hipblasLtMatrixLayoutSetAttribute(matA, HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
                                   &batch_count, sizeof(batch_count));
hipblasLtMatrixLayoutSetAttribute(matA, HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
                                   &stride_a, sizeof(stride_a));
```

---

## 9. Supported Data Type Combinations

The GEMM equation: `D = Activation(alpha * op(A) * op(B) + beta * op(C) + bias)`

### 9.1 Common Configurations

| A,B Type | C,D Type | Compute Type | GPU | Notes |
|----------|----------|--------------|-----|-------|
| f16_r | f16_r | f32_r | all | Standard FP16 |
| bf16_r | bf16_r | f32_r | all | Standard BF16 |
| f32_r | f32_r | f32_r | all | FP32 |
| f8_fnuz_r | f8_fnuz_r | f32_r | gfx942 | MI300 FP8 |
| f8_r | f8_r | f32_r | gfx950 | MI350 FP8 (OCP) |
| f8_r | f16_r | f32_r | gfx950 | Mixed FP8/FP16 |
| i8_r | i8_r | i32_r | all | INT8 |
| f4_r | f4_r | f32_r | gfx950 | FP4 with block scaling |
| f32_r | f32_r | xf32_r | gfx942,gfx950 | TF32 |

### 9.2 Constraint: c_type must equal d_type

The library requires `--c_type` and `--d_type` to be identical.

---

## 10. Comparing Against Baselines

### 10.1 hipBLASLt vs rocBLAS

rocBLAS uses `rocblas-bench` with similar syntax:
```bash
# hipBLASLt
./hipblaslt-bench -m 4096 -n 4096 -k 4096 -r bf16_r -i 100

# rocBLAS (separate install)
rocblas-bench -f gemm -r bf16_r -m 4096 -n 4096 -k 4096 -i 100
```

Both report GFLOPS and us. Compare `hipblaslt-Gflops` vs `rocblas-Gflops`.

### 10.2 Automated Performance Suite

hipBLASLt ships `hipblaslt-perf` (Python script) for systematic benchmarking:
```bash
# Located at: clients/scripts/performance/hipblaslt-perf
./hipblaslt-perf --help
```

### 10.3 Sequence Benchmarking

For multi-kernel pipeline simulation (e.g., transformer layer), use `hipblaslt-sequence` with a YAML config:

```yaml
GeneralSettings:
    PrintKernelInfo: true
    Rotating: 512
    ColdIter: 1000
    Iter: 10
Layers:
    - LayerType: GEMM
      Size: [256, 256, 256, 1]   # [M, N, K, batch]
      Alpha: 1.0
      Beta: 0
      TransposeA: false
      TransposeB: false
      DataTypeA: f16_r
      DataTypeB: f16_r
      DataTypeC: f16_r
      DataTypeD: f16_r
      ComputeType: f32_r
      Epilogue: HIPBLASLT_EPILOGUE_DEFAULT
      AlgoIndex: 35023
    - LayerType: FLUSH
```

---

## 11. Common Pitfalls

1. **c_type != d_type**: Will throw error. Always set both or rely on `--precision`.
2. **OCP vs FNUZ on gfx942**: hipblaslt-bench auto-converts, but API code must use `_FNUZ` types on MI300.
3. **Split-K requires api_method mix or cpp**: Won't work with `--api_method c`.
4. **Swizzle requires TN layout**: `--transA T --transB N` and FP16/BF16/FP8 types.
5. **Tuning file version mismatch**: Tuning indices are tied to the library git version. The bench will reject mismatched tuning files.
6. **Small M with host timer**: Use `--use_gpu_timer` for M<64 to avoid host-side timing overhead.
7. **Cache warming**: For production measurements, use `--rotating 512` (exceeds L2) and `--cold_iters 20+`.
8. **Default workspace**: 128 MB. Some solutions need more; increase with `--workspace`.

---

## 12. Key Source Files Reference

| File | Purpose |
|------|---------|
| `clients/bench/src/client.cpp` | Main bench entry point, all CLI parsing |
| `clients/bench/include/program_options.hpp` | CLI option parser |
| `clients/common/include/testing_matmul.hpp` | Core GEMM benchmark/test implementation |
| `clients/common/include/argument_model.hpp` | Output formatting (CSV columns) |
| `clients/common/include/hipblaslt_arguments.hpp` | Arguments struct definition |
| `clients/common/src/efficiency_monitor.cpp` | Frequency/efficiency monitoring |
| `clients/samples/` | 27+ sample programs covering all API patterns |
| `docs/reference/env-variables.rst` | Environment variable reference |
| `docs/how-to/how-to-use-hipblaslt-offline-tuning.rst` | Offline tuning guide |
| `docs/how-to/how-to-use-streamk.rst` | Stream-K configuration guide |
| `CMakeLists.txt` | Root build configuration |
| `install.sh` | Build automation script |
